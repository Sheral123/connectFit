from app.services.ai_utils import generate_text

def generate_personalized_plan(preferences: dict) -> str:
    # Static context that simulates retrieval of related documents.
    context = (
        "Meal Plan Context: A balanced meal plan should include lean proteins, vegetables, whole grains, and healthy fats. "
        "Workout Plan Context: A comprehensive workout plan should incorporate cardio, strength training, and flexibility exercises."
    )
    prompt = (
        f"User Preferences: {preferences}\n\n"
        f"{context}\n\n"
        "Based on the above, generate a detailed weekly personalized meal and workout plan."
    )
    return generate_text(prompt)
