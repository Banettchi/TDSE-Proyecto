"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, List
from datetime import datetime


# ── Auth ────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    tenant_name: str
    user_name: str
    role: str


class UserCreate(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "doctor"
    specialty: Optional[str] = None
    medical_license: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: str
    specialty: Optional[str] = None
    tenant_id: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Tenant ──────────────────────────────────────────────────────
class TenantCreate(BaseModel):
    name: str
    type: str  # hospital, eps, clinica
    nit: Optional[str] = None
    city: Optional[str] = None
    department: Optional[str] = None
    default_model: str = "efficientnet"
    risk_threshold: float = 0.5
    fhir_endpoint: Optional[str] = None


class TenantResponse(BaseModel):
    id: str
    name: str
    type: str
    nit: Optional[str] = None
    city: Optional[str] = None
    default_model: str
    risk_threshold: float
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TenantConfig(BaseModel):
    default_model: Optional[str] = None
    risk_threshold: Optional[float] = None
    fhir_endpoint: Optional[str] = None
    max_requests_per_minute: Optional[int] = None


# ── Patient ─────────────────────────────────────────────────────
class PatientCreate(BaseModel):
    document_type: Optional[str] = "CC"
    document_number: Optional[str] = None
    full_name: str
    birth_date: Optional[datetime] = None
    gender: Optional[str] = None
    fitzpatrick_type: Optional[int] = Field(None, ge=1, le=6)
    city: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class PatientResponse(BaseModel):
    id: str
    full_name: str
    document_type: Optional[str] = None
    document_number: Optional[str] = None
    birth_date: Optional[datetime] = None
    gender: Optional[str] = None
    fitzpatrick_type: Optional[int] = None
    city: Optional[str] = None
    tenant_id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Analysis ────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    patient_id: str
    model_preference: Optional[str] = None  # efficientnet, resnet, vit
    image_source: str = "smartphone"  # smartphone, dermatoscope
    notes: Optional[str] = None


class ABCDEResult(BaseModel):
    asymmetry: float = Field(description="Asymmetry score 0-1")
    border: float = Field(description="Border irregularity score 0-1")
    color: float = Field(description="Color variance score 0-1")
    diameter: float = Field(description="Estimated diameter in mm")
    evolution: Optional[float] = Field(None, description="Evolution score 0-1 (requires history)")
    total_score: float = Field(description="Weighted total ABCDE score")


class AnalysisResponse(BaseModel):
    id: str
    patient_id: str
    patient_name: str
    model_used: str
    predicted_class: int
    predicted_class_name: str
    confidence: float
    risk_level: str
    is_malignant: bool
    class_probabilities: Dict[str, float]
    abcde_scores: Optional[ABCDEResult] = None
    inference_time_ms: float
    gradcam_image_path: Optional[str] = None
    fhir_observation_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Dashboard ───────────────────────────────────────────────────
class DashboardStats(BaseModel):
    total_analyses: int
    total_patients: int
    high_risk_count: int
    malignant_count: int
    avg_confidence: float
    model_distribution: Dict[str, int]
    risk_distribution: Dict[str, int]
    recent_analyses: List[dict]
    monthly_trend: List[dict]


class EquityReport(BaseModel):
    model_name: str
    fitzpatrick_group: str
    auc_roc: float
    sensitivity: float
    specificity: float
    total_samples: int
    gap_from_baseline: Optional[float] = None
    alert: bool = False


class ModelMetrics(BaseModel):
    model_name: str
    total_inferences: int
    avg_confidence: float
    avg_inference_time_ms: float
    accuracy_estimate: Optional[float] = None
