from app.services.ai_utils import generate_text

def get_nutrition_advice(query: str) -> str:
    prompt = (
        f"User Query: {query}\n\n"
        "Provide clear, concise, and evidence-based nutritional advice suitable for someone seeking a healthier lifestyle."
    )
    return generate_text(prompt)
