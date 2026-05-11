"""
HL7 FHIR R5 resource generation.
Encapsulates analysis results as FHIR Observation resources
for interoperability with institutional HIS/EHR systems.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional


class FHIRGenerator:
    """
    Generates HL7 FHIR R5 Observation resources from analysis results.
    Includes extensions for confidence level and model architecture used.
    """

    FHIR_BASE = "https://dermai.health/fhir"
    SYSTEM_LOINC = "http://loinc.org"
    SYSTEM_SNOMED = "http://snomed.info/sct"

    # SNOMED codes for skin lesion types
    SNOMED_CODES = {
        "Queratosis actínica (akiec)": {"code": "201101007", "display": "Actinic keratosis"},
        "Carcinoma basocelular (bcc)": {"code": "254701007", "display": "Basal cell carcinoma"},
        "Queratosis benigna (bkl)": {"code": "400083001", "display": "Benign keratosis"},
        "Dermatofibroma (df)": {"code": "109836006", "display": "Dermatofibroma"},
        "Melanoma (mel)": {"code": "372244006", "display": "Malignant melanoma"},
        "Nevo melanocítico (nv)": {"code": "398943008", "display": "Melanocytic nevus"},
        "Lesión vascular (vasc)": {"code": "400210000", "display": "Vascular lesion"},
    }

    RISK_CODES = {
        "bajo": {"code": "low", "display": "Low Risk"},
        "medio": {"code": "moderate", "display": "Moderate Risk"},
        "alto": {"code": "high", "display": "High Risk"},
        "muy_alto": {"code": "very-high", "display": "Very High Risk"},
    }

    @classmethod
    def generate_observation(cls,
                              analysis_result: dict,
                              patient_id: str,
                              practitioner_id: str,
                              tenant_id: str) -> dict:
        """
        Generate a FHIR R5 Observation resource from analysis results.

        Follows HL7 FHIR R5 specification with custom extensions
        for AI confidence and model architecture.
        """
        observation_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"

        predicted_class = analysis_result.get("predicted_class_name", "Unknown")
        snomed = cls.SNOMED_CODES.get(predicted_class, {
            "code": "106076001", "display": "Skin lesion"
        })
        risk = cls.RISK_CODES.get(
            analysis_result.get("risk_level", "medio"),
            {"code": "moderate", "display": "Moderate Risk"}
        )

        observation = {
            "resourceType": "Observation",
            "id": observation_id,
            "meta": {
                "profile": [f"{cls.FHIR_BASE}/StructureDefinition/DermAIObservation"],
                "versionId": "1",
                "lastUpdated": now
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "imaging",
                            "display": "Imaging"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": cls.SYSTEM_LOINC,
                        "code": "76654-5",
                        "display": "Skin lesion assessment"
                    }
                ],
                "text": "Evaluación de lesión cutánea mediante IA"
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
                "type": "Patient"
            },
            "performer": [
                {
                    "reference": f"Practitioner/{practitioner_id}",
                    "type": "Practitioner"
                }
            ],
            "effectiveDateTime": now,
            "issued": now,
            "valueCodeableConcept": {
                "coding": [
                    {
                        "system": cls.SYSTEM_SNOMED,
                        "code": snomed["code"],
                        "display": snomed["display"]
                    }
                ],
                "text": predicted_class
            },
            # ── Extensions (model architecture, confidence) ──
            "extension": [
                {
                    "url": f"{cls.FHIR_BASE}/Extension/ai-confidence",
                    "valueDecimal": analysis_result.get("confidence", 0)
                },
                {
                    "url": f"{cls.FHIR_BASE}/Extension/ai-model-architecture",
                    "valueString": analysis_result.get("model_used", "unknown")
                },
                {
                    "url": f"{cls.FHIR_BASE}/Extension/ai-model-display-name",
                    "valueString": analysis_result.get("model_display_name", "Unknown")
                },
                {
                    "url": f"{cls.FHIR_BASE}/Extension/risk-level",
                    "valueCode": risk["code"]
                },
                {
                    "url": f"{cls.FHIR_BASE}/Extension/inference-time-ms",
                    "valueDecimal": analysis_result.get("inference_time_ms", 0)
                },
                {
                    "url": f"{cls.FHIR_BASE}/Extension/tenant-id",
                    "valueString": tenant_id
                }
            ],
            # ── Interpretation ──
            "interpretation": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                            "code": "A" if analysis_result.get("is_malignant") else "N",
                            "display": "Abnormal" if analysis_result.get("is_malignant") else "Normal"
                        }
                    ],
                    "text": f"Riesgo {risk['display']} - {predicted_class}"
                }
            ],
            # ── Components (ABCDE scores) ──
            "component": cls._build_abcde_components(analysis_result),
            # ── Note ──
            "note": [
                {
                    "text": (
                        f"Análisis automatizado por DermAI. "
                        f"Modelo: {analysis_result.get('model_display_name', 'N/A')}. "
                        f"Razón de selección: {analysis_result.get('model_selection_reason', 'N/A')}. "
                        f"Este resultado es un apoyo al diagnóstico clínico y no reemplaza "
                        f"la evaluación de un dermatólogo certificado."
                    )
                }
            ]
        }

        return observation

    @classmethod
    def _build_abcde_components(cls, analysis_result: dict) -> list:
        """Build FHIR Observation components from ABCDE scores."""
        components = []
        abcde = analysis_result.get("abcde_scores")

        if not abcde:
            return components

        abcde_map = [
            ("A", "Asimetría", abcde.get("asymmetry")),
            ("B", "Bordes", abcde.get("border")),
            ("C", "Color", abcde.get("color")),
            ("D", "Diámetro (mm)", abcde.get("diameter")),
            ("E", "Evolución", abcde.get("evolution")),
        ]

        for code, display, value in abcde_map:
            if value is not None:
                components.append({
                    "code": {
                        "coding": [{
                            "system": f"{cls.FHIR_BASE}/CodeSystem/abcde",
                            "code": code,
                            "display": display
                        }]
                    },
                    "valueQuantity": {
                        "value": value,
                        "unit": "mm" if code == "D" else "score",
                        "system": "http://unitsofmeasure.org"
                    }
                })

        # Total ABCDE score
        total = abcde.get("total_score")
        if total is not None:
            components.append({
                "code": {
                    "coding": [{
                        "system": f"{cls.FHIR_BASE}/CodeSystem/abcde",
                        "code": "TOTAL",
                        "display": "Puntaje ABCDE Total"
                    }]
                },
                "valueQuantity": {
                    "value": total,
                    "unit": "score",
                    "system": "http://unitsofmeasure.org"
                }
            })

        return components

    @classmethod
    def validate_resource(cls, resource: dict) -> dict:
        """Basic validation of the generated FHIR resource."""
        required_fields = ["resourceType", "id", "status", "code", "subject"]
        errors = []

        for field in required_fields:
            if field not in resource:
                errors.append(f"Missing required field: {field}")

        if resource.get("resourceType") != "Observation":
            errors.append("resourceType must be 'Observation'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "resource_id": resource.get("id")
        }
