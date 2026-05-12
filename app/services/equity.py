from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models.entities import Analysis, Patient

class EquityMonitor:
    @staticmethod
    async def compute_metrics(db: AsyncSession, tenant_id: str) -> dict:
        """
        Computes equity metrics based on Fitzpatrick skin type.
        """
        # Join Analysis and Patient to get fitzpatrick_type for each analysis
        query = (
            select(
                Patient.fitzpatrick_type,
                func.count(Analysis.id).label("total_inferences"),
                func.avg(Analysis.confidence).label("avg_confidence")
            )
            .join(Patient, Analysis.patient_id == Patient.id)
            .where(Analysis.tenant_id == tenant_id)
            .group_by(Patient.fitzpatrick_type)
        )
        
        result = await db.execute(query)
        rows = result.all()
        
        metrics = []
        for fitz_type, total, avg_conf in rows:
            metrics.append({
                "fitzpatrick_type": fitz_type or "Unknown",
                "total_inferences": total,
                "avg_confidence": round(avg_conf or 0, 4)
            })
            
        return {"metrics": metrics}

    @staticmethod
    def generate_equity_report(metrics: dict) -> dict:
        """
        Generates an equity report from computed metrics.
        """
        return {
            "title": "Equity Report by Fitzpatrick Skin Type",
            "data": metrics.get("metrics", [])
        }
