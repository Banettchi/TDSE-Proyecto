import uuid
from datetime import datetime

class FHIRGenerator:
    @staticmethod
    def generate_observation(analysis_result: dict, patient_id: str, practitioner_id: str, tenant_id: str) -> dict:
        """
        Generates a simplified FHIR R5 Observation resource for a skin lesion analysis.
        """
        return {
            "resourceType": "Observation",
            "id": f"obs-{uuid.uuid4()}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "imaging",
                    "display": "Imaging"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "50984-4",
                    "display": "Skin lesion type"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "performer": [{"reference": f"Practitioner/{practitioner_id}"}],
            "effectiveDateTime": datetime.utcnow().isoformat() + "Z",
            "valueString": analysis_result.get("predicted_class_name", "Unknown"),
            "component": [
                {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "80261-1",
                            "display": "Malignant"
                        }]
                    },
                    "valueBoolean": analysis_result.get("is_malignant", False)
                },
                {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": "73719-7",
                            "display": "Risk level"
                        }]
                    },
                    "valueString": analysis_result.get("risk_level", "Unknown")
                }
            ],
            "note": [{
                "text": f"Model: {analysis_result.get('model_used')}, Confidence: {analysis_result.get('confidence')}"
            }]
        }

    @staticmethod
    def validate_resource(resource: dict) -> dict:
        """
        Validates the FHIR resource structurally.
        In this dummy implementation, we assume it's always valid.
        """
        return {
            "valid": True,
            "issues": []
        }
