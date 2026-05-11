"""
DermAI — Plataforma Institucional de Screening Dermatológico Multitenante
Main application entry point.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.ml.inference import inference_engine
from app.api.routes import (
    auth_router, tenants_router, users_router,
    patients_router, analysis_router, dashboard_router
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: init DB and load ML models."""
    print("=" * 60)
    print(f"  [DermAI] {settings.APP_NAME}")
    print(f"  v{settings.APP_VERSION}")
    print("=" * 60)

    # Initialize database
    print("\n[DB] Initializing database...")
    await init_db()
    print("  OK - Database ready")

    # Seed initial data if empty
    await seed_initial_data()

    # Load ML models
    print("\n[ML] Loading ML models...")
    try:
        inference_engine.load_all_models()
    except Exception as e:
        print(f"  WARN - Model loading: {e}")
        print("  INFO - Models will be loaded on first request")

    print(f"\n[OK] Platform ready at http://localhost:8000")
    print("=" * 60)

    yield

    print("\n[STOP] Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Plataforma institucional de screening dermatológico multitenante "
        "con motor de inferencia multimodelo (EfficientNet-B7, ResNet-50, ViT-Base), "
        "análisis ABCDE, Grad-CAM y conformidad HL7 FHIR R5."
    ),
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ──────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(tenants_router)
app.include_router(users_router)
app.include_router(patients_router)
app.include_router(analysis_router)
app.include_router(dashboard_router)

# ── Static files ────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
uploads_dir = settings.UPLOAD_DIR

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

if os.path.exists(uploads_dir):
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")


# ── Frontend routes ─────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(frontend_dir, "dashboard.html"))


@app.get("/analysis")
async def serve_analysis():
    return FileResponse(os.path.join(frontend_dir, "analysis.html"))


@app.get("/patients")
async def serve_patients():
    return FileResponse(os.path.join(frontend_dir, "patients.html"))


@app.get("/models-info")
async def serve_models():
    return FileResponse(os.path.join(frontend_dir, "models.html"))


@app.get("/admin")
async def serve_admin():
    return FileResponse(os.path.join(frontend_dir, "admin.html"))


# ── Health check ────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "models_loaded": list(inference_engine.models.keys()),
        "device": inference_engine.device
    }


@app.get("/api/models/available")
async def available_models():
    return {"models": inference_engine.get_available_models()}


# ── Error handlers ──────────────────────────────────────────────
@app.exception_handler(404)
async def not_found(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content={"detail": "Recurso no encontrado"})
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.exception_handler(500)
async def server_error(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Error interno del servidor"}
    )


# ── Seed Data ───────────────────────────────────────────────────
async def seed_initial_data():
    """Create demo tenant and users if database is empty."""
    from app.database import AsyncSessionLocal
    from app.models.entities import Tenant, User, Patient
    from app.services.auth import hash_password
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Tenant).limit(1))
        if result.scalar_one_or_none():
            print("  INFO - Database already seeded")
            return

        print("  [SEED] Seeding demo data...")

        # Demo tenants
        hospital = Tenant(
            name="Hospital Universitario San Ignacio",
            type="hospital",
            nit="860013570",
            city="Bogotá",
            department="Cundinamarca",
            default_model="efficientnet",
            risk_threshold=0.5
        )
        eps = Tenant(
            name="EPS Sanitas",
            type="eps",
            nit="800251440",
            city="Bogotá",
            department="Cundinamarca",
            default_model="resnet"
        )
        clinica = Tenant(
            name="Clínica Dermacenter",
            type="clinica",
            city="Medellín",
            department="Antioquia",
            default_model="vit"
        )

        db.add_all([hospital, eps, clinica])
        await db.flush()

        # Demo users (password: demo123)
        pwd = hash_password("demo123")
        users = [
            User(tenant_id=hospital.id, email="admin@hospital.co", hashed_password=pwd,
                 full_name="Dr. Carlos Ramírez", role="admin", specialty="Dermatología"),
            User(tenant_id=hospital.id, email="doctor@hospital.co", hashed_password=pwd,
                 full_name="Dra. María López", role="doctor", specialty="Dermatología",
                 medical_license="MED-2024-0891"),
            User(tenant_id=eps.id, email="admin@sanitas.co", hashed_password=pwd,
                 full_name="Dr. Andrés García", role="admin"),
            User(tenant_id=clinica.id, email="admin@dermacenter.co", hashed_password=pwd,
                 full_name="Dra. Laura Martínez", role="admin", specialty="Oncología cutánea"),
        ]
        db.add_all(users)
        await db.flush()

        # Demo patients
        patients = [
            Patient(tenant_id=hospital.id, full_name="Juan Pérez",
                    document_type="CC", document_number="1023456789",
                    gender="Masculino", fitzpatrick_type=3, city="Bogotá"),
            Patient(tenant_id=hospital.id, full_name="Ana Rodríguez",
                    document_type="CC", document_number="1098765432",
                    gender="Femenino", fitzpatrick_type=4, city="Cali"),
            Patient(tenant_id=hospital.id, full_name="Carlos Mendoza",
                    document_type="CC", document_number="1076543210",
                    gender="Masculino", fitzpatrick_type=5, city="Cartagena"),
            Patient(tenant_id=eps.id, full_name="María Fernanda Torres",
                    document_type="CC", document_number="1045678901",
                    gender="Femenino", fitzpatrick_type=2, city="Bogotá"),
        ]
        db.add_all(patients)
        await db.commit()

        print(f"  OK - Created {len([hospital, eps, clinica])} tenants, {len(users)} users, {len(patients)} patients")
        print(f"  INFO - Demo login: admin@hospital.co / demo123")
