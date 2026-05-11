"""
Database ORM models for the multi-tenant melanoma screening platform.
All models include tenant_id for data isolation.
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship
from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


# ── Tenant ──────────────────────────────────────────────────────
class Tenant(Base):
    __tablename__ = "tenants"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False)
    type = Column(String(50), nullable=False)  # hospital, eps, clinica, secretaria
    nit = Column(String(20), unique=True)  # Tax ID
    city = Column(String(100))
    department = Column(String(100))

    # Configuration
    default_model = Column(String(50), default="efficientnet")
    risk_threshold = Column(Float, default=0.5)
    fhir_endpoint = Column(String(500))  # HIS/EHR integration URL
    max_requests_per_minute = Column(Integer, default=60)

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    patients = relationship("Patient", back_populates="tenant", cascade="all, delete-orphan")
    analyses = relationship("Analysis", back_populates="tenant", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="tenant", cascade="all, delete-orphan")


# ── User ────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    full_name = Column(String(200), nullable=False)
    role = Column(String(50), default="doctor")  # admin, doctor, operator
    specialty = Column(String(100))
    medical_license = Column(String(50))

    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    analyses = relationship("Analysis", back_populates="created_by_user")


# ── Patient ─────────────────────────────────────────────────────
class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    document_type = Column(String(20))  # CC, TI, CE, etc.
    document_number = Column(String(30))
    full_name = Column(String(200), nullable=False)
    birth_date = Column(DateTime)
    gender = Column(String(20))
    fitzpatrick_type = Column(Integer)  # 1-6 (Fitzpatrick skin type)
    city = Column(String(100))
    phone = Column(String(20))
    email = Column(String(200))

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    tenant = relationship("Tenant", back_populates="patients")
    analyses = relationship("Analysis", back_populates="patient", cascade="all, delete-orphan")


# ── Analysis ────────────────────────────────────────────────────
class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    created_by = Column(String, ForeignKey("users.id"), nullable=False)

    # Image data
    original_image_path = Column(String(500), nullable=False)
    processed_image_path = Column(String(500))
    gradcam_image_path = Column(String(500))
    image_source = Column(String(50), default="smartphone")  # smartphone, dermatoscope
    image_quality_score = Column(Float)

    # Model info
    model_used = Column(String(50), nullable=False)  # efficientnet, resnet, vit
    model_version = Column(String(50))

    # Classification results
    predicted_class = Column(Integer)
    predicted_class_name = Column(String(100))
    confidence = Column(Float)
    risk_level = Column(String(20))  # bajo, medio, alto, muy_alto
    is_malignant = Column(Boolean)
    class_probabilities = Column(JSON)  # {class_name: probability}

    # ABCDE Analysis
    abcde_scores = Column(JSON)  # {A: score, B: score, C: score, D: score, E: score}
    abcde_total = Column(Float)
    asymmetry_score = Column(Float)
    border_score = Column(Float)
    color_score = Column(Float)
    diameter_mm = Column(Float)
    evolution_score = Column(Float)

    # FHIR
    fhir_observation_id = Column(String(100))
    fhir_resource = Column(JSON)

    # Metadata
    inference_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)

    # Relationships
    tenant = relationship("Tenant", back_populates="analyses")
    patient = relationship("Patient", back_populates="analyses")
    created_by_user = relationship("User", back_populates="analyses")


# ── Audit Log ───────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(String)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String)
    details = Column(JSON)
    ip_address = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)

    tenant = relationship("Tenant", back_populates="audit_logs")


# ── Equity Metrics ──────────────────────────────────────────────
class EquityMetric(Base):
    __tablename__ = "equity_metrics"

    id = Column(String, primary_key=True, default=generate_uuid)
    tenant_id = Column(String)
    model_name = Column(String(50), nullable=False)
    fitzpatrick_group = Column(String(20), nullable=False)  # I-III, IV-VI
    auc_roc = Column(Float)
    sensitivity = Column(Float)
    specificity = Column(Float)
    total_samples = Column(Integer)
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    alert_triggered = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
