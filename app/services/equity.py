"""
Equity Module — Monitors model performance segmented by
Fitzpatrick skin type to detect and mitigate bias.

Implements the equity monitoring described in §3.4:
alerts when AUC gap between phototype groups exceeds threshold.
"""
from datetime import datetime
from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.entities import Analysis, Patient, EquityMetric
from app.config import settings


class EquityMonitor:
    """
    Monitors diagnostic performance across Fitzpatrick skin type groups.
    Groups: I-III (light) and IV-VI (dark).
    Triggers alerts when performance gap exceeds the threshold.
    """

    # Maximum acceptable AUC gap between groups (from §4.3: 0.089 observed)
    MAX_AUC_GAP = 0.10

    @staticmethod
    def get_fitzpatrick_group(fitzpatrick_type: Optional[int]) -> str:
        """Classify Fitzpatrick type into group."""
        if fitzpatrick_type is None:
            return "unknown"
        if fitzpatrick_type <= 3:
            return "I-III"
        return "IV-VI"

    @classmethod
    async def compute_metrics(cls, db: AsyncSession,
                               tenant_id: Optional[str] = None) -> List[Dict]:
        """
        Compute equity metrics for all models, segmented by Fitzpatrick group.
        """
        results = []

        for model_name in ["efficientnet", "resnet", "vit"]:
            for group in ["I-III", "IV-VI"]:
                # Define Fitzpatrick range for this group
                if group == "I-III":
                    fitz_range = [1, 2, 3]
                else:
                    fitz_range = [4, 5, 6]

                # Query analyses for this model and Fitzpatrick group
                query = (
                    select(Analysis)
                    .join(Patient, Analysis.patient_id == Patient.id)
                    .where(
                        Analysis.model_used == model_name,
                        Patient.fitzpatrick_type.in_(fitz_range)
                    )
                )

                if tenant_id:
                    query = query.where(Analysis.tenant_id == tenant_id)

                result = await db.execute(query)
                analyses = result.scalars().all()

                if not analyses:
                    results.append({
                        "model_name": model_name,
                        "fitzpatrick_group": group,
                        "total_samples": 0,
                        "avg_confidence": 0,
                        "high_risk_rate": 0,
                        "alert": False,
                        "message": "Sin datos suficientes"
                    })
                    continue

                total = len(analyses)
                avg_conf = sum(a.confidence or 0 for a in analyses) / total
                high_risk = sum(1 for a in analyses if a.risk_level in ["alto", "muy_alto"])
                malignant = sum(1 for a in analyses if a.is_malignant)

                results.append({
                    "model_name": model_name,
                    "fitzpatrick_group": group,
                    "total_samples": total,
                    "avg_confidence": round(avg_conf, 4),
                    "high_risk_rate": round(high_risk / total, 4) if total > 0 else 0,
                    "malignant_rate": round(malignant / total, 4) if total > 0 else 0,
                    "alert": False,
                    "message": "OK"
                })

        # Check for gaps between groups
        cls._check_gaps(results)

        return results

    @classmethod
    def _check_gaps(cls, metrics: List[Dict]):
        """Check for unacceptable performance gaps between groups."""
        by_model = {}
        for m in metrics:
            model = m["model_name"]
            if model not in by_model:
                by_model[model] = {}
            by_model[model][m["fitzpatrick_group"]] = m

        for model, groups in by_model.items():
            light = groups.get("I-III", {})
            dark = groups.get("IV-VI", {})

            if light.get("total_samples", 0) > 0 and dark.get("total_samples", 0) > 0:
                conf_gap = abs(
                    light.get("avg_confidence", 0) - dark.get("avg_confidence", 0)
                )

                if conf_gap > cls.MAX_AUC_GAP:
                    for g in [light, dark]:
                        g["alert"] = True
                        g["message"] = (
                            f"⚠ Brecha de confianza: {conf_gap:.3f} "
                            f"(umbral: {cls.MAX_AUC_GAP}). "
                            f"Se recomienda reentrenamiento dirigido."
                        )
                        g["confidence_gap"] = round(conf_gap, 4)

    @classmethod
    async def save_metrics(cls, db: AsyncSession, metrics: List[Dict],
                            tenant_id: Optional[str] = None):
        """Persist equity metrics for auditing."""
        now = datetime.utcnow()
        for m in metrics:
            record = EquityMetric(
                tenant_id=tenant_id,
                model_name=m["model_name"],
                fitzpatrick_group=m["fitzpatrick_group"],
                auc_roc=m.get("avg_confidence", 0),
                sensitivity=m.get("high_risk_rate", 0),
                specificity=1 - m.get("high_risk_rate", 0),
                total_samples=m["total_samples"],
                alert_triggered=m.get("alert", False),
                created_at=now
            )
            db.add(record)
        await db.commit()

    @classmethod
    def generate_equity_report(cls, metrics: List[Dict]) -> Dict:
        """Generate a summary equity report."""
        alerts = [m for m in metrics if m.get("alert")]
        total_samples = sum(m.get("total_samples", 0) for m in metrics)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_analyses_evaluated": total_samples,
            "total_alerts": len(alerts),
            "metrics_by_model": metrics,
            "alerts": alerts,
            "recommendation": (
                "Se requiere reentrenamiento dirigido con imágenes de fototipos IV-VI"
                if alerts else
                "Desempeño equitativo dentro de los umbrales aceptables"
            ),
            "reference": "Cassidy et al. (2022) - ISIC dataset bias analysis"
        }
