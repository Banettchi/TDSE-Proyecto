class ModelRouter:
    @staticmethod
    def select_model(
        user_override: str = None,
        tenant_default: str = "efficientnet",
        image_source: str = "smartphone",
        quality_score: float = 1.0,
        fitzpatrick_type: int = None
    ) -> dict:
        """
        Selects the most appropriate model based on input parameters.
        """
        if user_override:
            return {
                "model": user_override,
                "reason": "User override selected."
            }

        if image_source == "dermatoscope" or quality_score > 0.8:
            return {
                "model": tenant_default,
                "reason": f"High quality image/dermatoscope -> using tenant default ({tenant_default})."
            }
        else:
            return {
                "model": tenant_default,
                "reason": f"Using tenant default ({tenant_default}) as fallback."
            }
