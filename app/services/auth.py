"""
Authentication service — JWT-based auth with tenant isolation.
"""
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status

from app.config import settings
from app.models.entities import User, Tenant


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT token with tenant and role claims."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def authenticate_user(db: AsyncSession, email: str, password: str) -> Optional[User]:
    """Authenticate user by email and password."""
    result = await db.execute(select(User).where(User.email == email, User.is_active == True))
    user = result.scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        return None

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    return user


async def get_user_with_tenant(db: AsyncSession, user_id: str) -> Optional[tuple]:
    """Get user and their tenant."""
    result = await db.execute(
        select(User, Tenant)
        .join(Tenant, User.tenant_id == Tenant.id)
        .where(User.id == user_id, User.is_active == True)
    )
    row = result.first()
    if row:
        return row[0], row[1]
    return None


async def create_user(db: AsyncSession, tenant_id: str, email: str,
                      password: str, full_name: str, role: str = "doctor",
                      specialty: str = None, medical_license: str = None) -> User:
    """Create a new user in a specific tenant."""
    # Check if email exists
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="El email ya está registrado")

    user = User(
        tenant_id=tenant_id,
        email=email,
        hashed_password=hash_password(password),
        full_name=full_name,
        role=role,
        specialty=specialty,
        medical_license=medical_license
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
