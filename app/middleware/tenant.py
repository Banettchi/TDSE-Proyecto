"""
Tenant isolation middleware and authentication dependencies.
Ensures every request is scoped to the authenticated tenant.
"""
import time
from collections import defaultdict
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.auth import decode_token, get_user_with_tenant
from app.models.entities import AuditLog

security = HTTPBearer(auto_error=False)

# ── Rate Limiting (in-memory for prototype) ─────────────────────
_rate_limits: dict = defaultdict(list)
RATE_WINDOW = 60  # seconds


def _check_rate_limit(tenant_id: str, max_requests: int = 60):
    """Simple in-memory rate limiter per tenant."""
    now = time.time()
    window_start = now - RATE_WINDOW
    
    # Clean old entries
    _rate_limits[tenant_id] = [
        t for t in _rate_limits[tenant_id] if t > window_start
    ]
    
    if len(_rate_limits[tenant_id]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit excedido ({max_requests} req/min). Intente más tarde."
        )
    
    _rate_limits[tenant_id].append(now)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """
    Dependency: Extract and validate the current user from JWT.
    Returns (user, tenant) tuple for tenant-scoped operations.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de autenticación requerido",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = decode_token(credentials.credentials)
    user_id = payload.get("sub")
    
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido: sin identificador de usuario"
        )

    result = await get_user_with_tenant(db, user_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario no encontrado o desactivado"
        )

    user, tenant = result

    if not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="El tenant está desactivado"
        )

    # Rate limiting per tenant
    _check_rate_limit(tenant.id, tenant.max_requests_per_minute)

    return user, tenant


def require_role(*roles: str):
    """Dependency factory: require specific role(s)."""
    async def checker(auth=Depends(get_current_user)):
        user, tenant = auth
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rol '{user.role}' no autorizado. Se requiere: {', '.join(roles)}"
            )
        return user, tenant
    return checker


async def log_audit(db: AsyncSession, tenant_id: str, user_id: str,
                    action: str, resource_type: str = None,
                    resource_id: str = None, details: dict = None,
                    ip_address: str = None):
    """Write an immutable audit log entry."""
    entry = AuditLog(
        tenant_id=tenant_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details,
        ip_address=ip_address
    )
    db.add(entry)
    await db.commit()
