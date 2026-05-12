"""
API Endpoints — All REST routes for the platform.
Organized by domain: auth, tenants, patients, analysis, dashboard.
"""
import os
import uuid
import shutil
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.database import get_db
from app.config import settings
from app.models.entities import Tenant, User, Patient, Analysis, AuditLog, EquityMetric
from app.schemas.schemas import (
    LoginRequest, TokenResponse, UserCreate, UserResponse,
    TenantCreate, TenantResponse, TenantConfig,
    PatientCreate, PatientResponse,
    AnalysisResponse, DashboardStats
)
from app.services.auth import (
    authenticate_user, create_access_token, create_user, hash_password
)
from app.services.fhir import FHIRGenerator
from app.services.equity import EquityMonitor
from app.middleware.tenant import get_current_user, require_role, log_audit
from app.ml.inference import inference_engine


# ════════════════════════════════════════════════════════════════
#  AUTH ROUTER
# ════════════════════════════════════════════════════════════════
auth_router = APIRouter(prefix="/api/auth", tags=["Autenticación"])


@auth_router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT token."""
    user = await authenticate_user(db, request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    # Get tenant
    result = await db.execute(select(Tenant).where(Tenant.id == user.tenant_id))
    tenant = result.scalar_one()

    token = create_access_token({
        "sub": user.id,
        "tenant_id": user.tenant_id,
        "role": user.role,
        "email": user.email
    })

    await log_audit(db, tenant.id, user.id, "LOGIN", "user", user.id)

    return TokenResponse(
        access_token=token,
        tenant_id=tenant.id,
        tenant_name=tenant.name,
        user_name=user.full_name,
        role=user.role
    )


@auth_router.get("/me", response_model=UserResponse)
async def get_me(auth=Depends(get_current_user)):
    """Get current user info."""
    user, tenant = auth
    return user


# ════════════════════════════════════════════════════════════════
#  TENANTS ROUTER
# ════════════════════════════════════════════════════════════════
tenants_router = APIRouter(prefix="/api/tenants", tags=["Tenants"])


@tenants_router.post("/", response_model=TenantResponse)
async def create_tenant(data: TenantCreate, db: AsyncSession = Depends(get_db)):
    """Create a new tenant (public endpoint for setup)."""
    tenant = Tenant(
        name=data.name,
        type=data.type,
        nit=data.nit,
        city=data.city,
        department=data.department,
        default_model=data.default_model,
        risk_threshold=data.risk_threshold,
        fhir_endpoint=data.fhir_endpoint
    )
    db.add(tenant)
    await db.commit()
    await db.refresh(tenant)
    return tenant


@tenants_router.get("/current", response_model=TenantResponse)
async def get_current_tenant(auth=Depends(get_current_user)):
    """Get current tenant info."""
    _, tenant = auth
    return tenant


@tenants_router.put("/config")
async def update_tenant_config(
    config: TenantConfig,
    auth=Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """Update tenant configuration (admin only)."""
    user, tenant = auth

    if config.default_model:
        tenant.default_model = config.default_model
    if config.risk_threshold is not None:
        tenant.risk_threshold = config.risk_threshold
    if config.fhir_endpoint is not None:
        tenant.fhir_endpoint = config.fhir_endpoint
    if config.max_requests_per_minute is not None:
        tenant.max_requests_per_minute = config.max_requests_per_minute

    await db.commit()
    await log_audit(db, tenant.id, user.id, "UPDATE_CONFIG", "tenant", tenant.id,
                    details=config.model_dump(exclude_none=True))

    return {"message": "Configuración actualizada", "tenant_id": tenant.id}


@tenants_router.post("/register")
async def register_tenant_with_admin(
    tenant_name: str = Form(...),
    tenant_type: str = Form(...),
    admin_email: str = Form(...),
    admin_password: str = Form(...),
    admin_name: str = Form(...),
    city: str = Form(None),
    nit: str = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Register a new tenant with an admin user."""
    # Create tenant
    tenant = Tenant(
        name=tenant_name,
        type=tenant_type,
        city=city,
        nit=nit
    )
    db.add(tenant)
    await db.flush()

    # Create admin user
    admin = User(
        tenant_id=tenant.id,
        email=admin_email,
        hashed_password=hash_password(admin_password),
        full_name=admin_name,
        role="admin"
    )
    db.add(admin)
    await db.commit()

    return {
        "message": "Tenant y administrador creados exitosamente",
        "tenant_id": tenant.id,
        "tenant_name": tenant.name,
        "admin_email": admin.email
    }


# ════════════════════════════════════════════════════════════════
#  USERS ROUTER
# ════════════════════════════════════════════════════════════════
users_router = APIRouter(prefix="/api/users", tags=["Usuarios"])


@users_router.post("/", response_model=UserResponse)
async def create_new_user(
    data: UserCreate,
    auth=Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """Create a new user in the current tenant (admin only)."""
    user, tenant = auth
    new_user = await create_user(
        db, tenant.id, data.email, data.password,
        data.full_name, data.role, data.specialty, data.medical_license
    )
    await log_audit(db, tenant.id, user.id, "CREATE_USER", "user", new_user.id)
    return new_user


@users_router.get("/", response_model=List[UserResponse])
async def list_users(
    auth=Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """List all users in the current tenant."""
    _, tenant = auth
    result = await db.execute(
        select(User).where(User.tenant_id == tenant.id).order_by(desc(User.created_at))
    )
    return result.scalars().all()


# ════════════════════════════════════════════════════════════════
#  PATIENTS ROUTER
# ════════════════════════════════════════════════════════════════
patients_router = APIRouter(prefix="/api/patients", tags=["Pacientes"])


@patients_router.post("/", response_model=PatientResponse)
async def create_patient(
    data: PatientCreate,
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new patient in the current tenant."""
    user, tenant = auth
    patient = Patient(
        tenant_id=tenant.id,
        **data.model_dump()
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)

    await log_audit(db, tenant.id, user.id, "CREATE_PATIENT", "patient", patient.id)
    return patient


@patients_router.get("/", response_model=List[PatientResponse])
async def list_patients(
    search: Optional[str] = Query(None),
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List patients in the current tenant (tenant-isolated)."""
    _, tenant = auth
    query = select(Patient).where(Patient.tenant_id == tenant.id)

    if search:
        query = query.where(Patient.full_name.ilike(f"%{search}%"))

    query = query.order_by(desc(Patient.created_at))
    result = await db.execute(query)
    return result.scalars().all()


@patients_router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: str,
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific patient (tenant-isolated)."""
    _, tenant = auth
    result = await db.execute(
        select(Patient).where(Patient.id == patient_id, Patient.tenant_id == tenant.id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(404, "Paciente no encontrado")
    return patient


@patients_router.get("/{patient_id}/history")
async def get_patient_history(
    patient_id: str,
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get analysis history for a patient (for Evolution tracking)."""
    _, tenant = auth
    result = await db.execute(
        select(Analysis)
        .where(Analysis.patient_id == patient_id, Analysis.tenant_id == tenant.id)
        .order_by(desc(Analysis.created_at))
    )
    analyses = result.scalars().all()

    return [{
        "id": a.id,
        "model_used": a.model_used,
        "predicted_class_name": a.predicted_class_name,
        "confidence": a.confidence,
        "risk_level": a.risk_level,
        "is_malignant": a.is_malignant,
        "abcde_scores": a.abcde_scores,
        "inference_time_ms": a.inference_time_ms,
        "created_at": a.created_at.isoformat() if a.created_at else None,
        "gradcam_image_path": a.gradcam_image_path
    } for a in analyses]


# ════════════════════════════════════════════════════════════════
#  ANALYSIS ROUTER (CORE)
# ════════════════════════════════════════════════════════════════
analysis_router = APIRouter(prefix="/api/analysis", tags=["Análisis"])


@analysis_router.post("/")
async def run_analysis(
    patient_id: str = Form(...),
    image: UploadFile = File(...),
    model_preference: Optional[str] = Form(None),
    image_source: str = Form("smartphone"),
    notes: Optional[str] = Form(None),
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Run a complete skin lesion analysis.
    This is the main endpoint that triggers the full pipeline:
    preprocessing → model selection → classification → Grad-CAM → ABCDE → FHIR
    """
    user, tenant = auth

    # Validate patient belongs to tenant
    result = await db.execute(
        select(Patient).where(Patient.id == patient_id, Patient.tenant_id == tenant.id)
    )
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(404, "Paciente no encontrado en este tenant")

    # Save uploaded image
    file_ext = os.path.splitext(image.filename)[1] or ".jpg"
    image_filename = f"{uuid.uuid4()}{file_ext}"
    image_path = os.path.join(settings.UPLOAD_DIR, image_filename)

    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Check for previous analysis (for evolution tracking)
    prev_result = await db.execute(
        select(Analysis)
        .where(Analysis.patient_id == patient_id, Analysis.tenant_id == tenant.id)
        .order_by(desc(Analysis.created_at))
        .limit(1)
    )
    prev_analysis = prev_result.scalar_one_or_none()
    prev_image_path = prev_analysis.original_image_path if prev_analysis else None

    # Run inference
    try:
        inference_result = await inference_engine.run_inference(
            image_path=image_path,
            model_name=model_preference,
            image_source=image_source,
            fitzpatrick_type=patient.fitzpatrick_type,
            tenant_default_model=tenant.default_model,
            previous_image_path=prev_image_path
        )
    except Exception as e:
        raise HTTPException(500, f"Error en inferencia: {str(e)}")

    # Generate FHIR Observation
    fhir_resource = FHIRGenerator.generate_observation(
        analysis_result=inference_result,
        patient_id=patient_id,
        practitioner_id=user.id,
        tenant_id=tenant.id
    )
    fhir_validation = FHIRGenerator.validate_resource(fhir_resource)

    # Convertir tipos numpy a tipos Python nativos
    import numpy as np
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    inference_result = convert_numpy(inference_result)
    fhir_resource = convert_numpy(fhir_resource)

    # Save analysis to database
    analysis = Analysis(
        tenant_id=tenant.id,
        patient_id=patient_id,
        created_by=user.id,
        original_image_path=image_path,
        processed_image_path=inference_result.get("processed_image_path"),
        gradcam_image_path=inference_result.get("gradcam_image_path"),
        image_source=image_source,
        image_quality_score=float(inference_result["image_quality_score"]) if inference_result.get("image_quality_score") is not None else None,
        model_used=inference_result["model_used"],
        model_version="1.0",
        predicted_class=int(inference_result["predicted_class"]),
        predicted_class_name=inference_result["predicted_class_name"],
        confidence=float(inference_result["confidence"]),
        risk_level=inference_result["risk_level"],
        is_malignant=bool(inference_result["is_malignant"]),
        class_probabilities={k: float(v) for k, v in inference_result["class_probabilities"].items()},
        abcde_scores=inference_result.get("abcde_scores"),
        abcde_total=float(inference_result["abcde_scores"]["total_score"]) if inference_result.get("abcde_scores") and inference_result["abcde_scores"].get("total_score") is not None else None,
        asymmetry_score=float(inference_result["abcde_scores"]["asymmetry"]) if inference_result.get("abcde_scores") and inference_result["abcde_scores"].get("asymmetry") is not None else None,
        border_score=float(inference_result["abcde_scores"]["border"]) if inference_result.get("abcde_scores") and inference_result["abcde_scores"].get("border") is not None else None,
        color_score=float(inference_result["abcde_scores"]["color"]) if inference_result.get("abcde_scores") and inference_result["abcde_scores"].get("color") is not None else None,
        diameter_mm=float(inference_result["abcde_scores"]["diameter"]) if inference_result.get("abcde_scores") and inference_result["abcde_scores"].get("diameter") is not None else None,
        evolution_score=float(inference_result["abcde_scores"]["evolution"]) if inference_result.get("abcde_scores") and inference_result["abcde_scores"].get("evolution") is not None else None,
        fhir_observation_id=fhir_resource.get("id"),
        fhir_resource=fhir_resource,
        inference_time_ms=float(inference_result["inference_time_ms"]),
        notes=notes
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    await log_audit(db, tenant.id, user.id, "RUN_ANALYSIS", "analysis", analysis.id,
                    details={"model": inference_result["model_used"],
                             "risk": inference_result["risk_level"]})

    return {
        "analysis_id": analysis.id,
        "patient_name": patient.full_name,
        **inference_result,
        "fhir_observation_id": fhir_resource.get("id"),
        "fhir_valid": fhir_validation["valid"]
    }


@analysis_router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific analysis result (tenant-isolated)."""
    _, tenant = auth
    result = await db.execute(
        select(Analysis, Patient)
        .join(Patient, Analysis.patient_id == Patient.id)
        .where(Analysis.id == analysis_id, Analysis.tenant_id == tenant.id)
    )
    row = result.first()
    if not row:
        raise HTTPException(404, "Análisis no encontrado")

    analysis, patient = row
    return {
        "id": analysis.id,
        "patient_id": patient.id,
        "patient_name": patient.full_name,
        "model_used": analysis.model_used,
        "predicted_class": analysis.predicted_class,
        "predicted_class_name": analysis.predicted_class_name,
        "confidence": analysis.confidence,
        "risk_level": analysis.risk_level,
        "is_malignant": analysis.is_malignant,
        "class_probabilities": analysis.class_probabilities,
        "abcde_scores": analysis.abcde_scores,
        "inference_time_ms": analysis.inference_time_ms,
        "image_quality_score": analysis.image_quality_score,
        "fhir_resource": analysis.fhir_resource,
        "notes": analysis.notes,
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "gradcam_available": analysis.gradcam_image_path is not None
    }


@analysis_router.get("/{analysis_id}/fhir")
async def get_fhir_resource(
    analysis_id: str,
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the FHIR R5 Observation resource for an analysis."""
    _, tenant = auth
    result = await db.execute(
        select(Analysis).where(Analysis.id == analysis_id, Analysis.tenant_id == tenant.id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(404, "Análisis no encontrado")

    return analysis.fhir_resource


@analysis_router.get("/")
async def list_analyses(
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    risk_level: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List analyses for the current tenant."""
    _, tenant = auth
    query = (
        select(Analysis, Patient)
        .join(Patient, Analysis.patient_id == Patient.id)
        .where(Analysis.tenant_id == tenant.id)
    )

    if risk_level:
        query = query.where(Analysis.risk_level == risk_level)
    if model:
        query = query.where(Analysis.model_used == model)

    query = query.order_by(desc(Analysis.created_at)).offset(offset).limit(limit)
    result = await db.execute(query)
    rows = result.all()

    # Count total
    count_query = select(func.count(Analysis.id)).where(Analysis.tenant_id == tenant.id)
    if risk_level:
        count_query = count_query.where(Analysis.risk_level == risk_level)
    count_result = await db.execute(count_query)
    total = count_result.scalar()

    return {
        "total": total,
        "analyses": [{
            "id": a.id,
            "patient_name": p.full_name,
            "patient_id": p.id,
            "model_used": a.model_used,
            "predicted_class_name": a.predicted_class_name,
            "confidence": a.confidence,
            "risk_level": a.risk_level,
            "is_malignant": a.is_malignant,
            "inference_time_ms": a.inference_time_ms,
            "created_at": a.created_at.isoformat() if a.created_at else None
        } for a, p in rows]
    }


# ════════════════════════════════════════════════════════════════
#  DASHBOARD ROUTER
# ════════════════════════════════════════════════════════════════
dashboard_router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@dashboard_router.get("/stats")
async def get_dashboard_stats(
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get dashboard statistics for the current tenant."""
    _, tenant = auth
    tid = tenant.id

    # Total analyses
    total_q = await db.execute(
        select(func.count(Analysis.id)).where(Analysis.tenant_id == tid)
    )
    total_analyses = total_q.scalar() or 0

    # Total patients
    patients_q = await db.execute(
        select(func.count(Patient.id)).where(Patient.tenant_id == tid)
    )
    total_patients = patients_q.scalar() or 0

    # Risk distribution
    risk_q = await db.execute(
        select(Analysis.risk_level, func.count(Analysis.id))
        .where(Analysis.tenant_id == tid)
        .group_by(Analysis.risk_level)
    )
    risk_dist = {r: c for r, c in risk_q.all() if r}

    # Model distribution
    model_q = await db.execute(
        select(Analysis.model_used, func.count(Analysis.id))
        .where(Analysis.tenant_id == tid)
        .group_by(Analysis.model_used)
    )
    model_dist = {m: c for m, c in model_q.all() if m}

    # Malignant count
    mal_q = await db.execute(
        select(func.count(Analysis.id))
        .where(Analysis.tenant_id == tid, Analysis.is_malignant == True)
    )
    malignant_count = mal_q.scalar() or 0

    # Average confidence
    conf_q = await db.execute(
        select(func.avg(Analysis.confidence)).where(Analysis.tenant_id == tid)
    )
    avg_confidence = round(conf_q.scalar() or 0, 4)

    # Average inference time
    time_q = await db.execute(
        select(func.avg(Analysis.inference_time_ms)).where(Analysis.tenant_id == tid)
    )
    avg_time = round(time_q.scalar() or 0, 2)

    # Recent analyses
    recent_q = await db.execute(
        select(Analysis, Patient)
        .join(Patient, Analysis.patient_id == Patient.id)
        .where(Analysis.tenant_id == tid)
        .order_by(desc(Analysis.created_at))
        .limit(10)
    )
    recent = [{
        "id": a.id,
        "patient_name": p.full_name,
        "model_used": a.model_used,
        "predicted_class_name": a.predicted_class_name,
        "risk_level": a.risk_level,
        "confidence": a.confidence,
        "created_at": a.created_at.isoformat() if a.created_at else None
    } for a, p in recent_q.all()]

    return {
        "total_analyses": total_analyses,
        "total_patients": total_patients,
        "malignant_count": malignant_count,
        "high_risk_count": risk_dist.get("alto", 0) + risk_dist.get("muy_alto", 0),
        "avg_confidence": avg_confidence,
        "avg_inference_time_ms": avg_time,
        "risk_distribution": risk_dist,
        "model_distribution": model_dist,
        "recent_analyses": recent,
        "tenant_name": tenant.name,
        "tenant_model": tenant.default_model
    }


@dashboard_router.get("/models")
async def get_model_metrics(
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get per-model performance metrics."""
    _, tenant = auth

    models_data = []
    for model_name in ["efficientnet", "resnet", "vit"]:
        q = await db.execute(
            select(
                func.count(Analysis.id),
                func.avg(Analysis.confidence),
                func.avg(Analysis.inference_time_ms)
            ).where(
                Analysis.tenant_id == tenant.id,
                Analysis.model_used == model_name
            )
        )
        row = q.first()
        count, avg_conf, avg_time = row if row else (0, 0, 0)

        from app.ml.models import MODEL_REGISTRY
        info = MODEL_REGISTRY.get(model_name, {})

        models_data.append({
            "name": model_name,
            "display_name": info.get("display_name", model_name),
            "total_inferences": count or 0,
            "avg_confidence": round(avg_conf or 0, 4),
            "avg_inference_time_ms": round(avg_time or 0, 2),
            "auc_range": info.get("auc_range", "N/A"),
            "context": info.get("context", ""),
            "is_default": model_name == tenant.default_model
        })

    return {"models": models_data}


@dashboard_router.get("/equity")
async def get_equity_report(
    auth=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get equity metrics by Fitzpatrick skin type."""
    _, tenant = auth

    metrics = await EquityMonitor.compute_metrics(db, tenant.id)
    report = EquityMonitor.generate_equity_report(metrics)

    return report


@dashboard_router.get("/audit")
async def get_audit_log(
    limit: int = Query(50, le=200),
    auth=Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """Get audit log for the current tenant (admin only)."""
    _, tenant = auth
    result = await db.execute(
        select(AuditLog)
        .where(AuditLog.tenant_id == tenant.id)
        .order_by(desc(AuditLog.timestamp))
        .limit(limit)
    )
    logs = result.scalars().all()

    return [{
        "id": l.id,
        "user_id": l.user_id,
        "action": l.action,
        "resource_type": l.resource_type,
        "resource_id": l.resource_id,
        "details": l.details,
        "timestamp": l.timestamp.isoformat() if l.timestamp else None
    } for l in logs]
