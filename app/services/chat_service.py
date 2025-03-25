from app.services.ai_utils import generate_text

def get_chat_response(message: str) -> str:
    # Construct a prompt to set the role and context for Gemini.
    prompt = (
        "You are a friendly and knowledgeable fitness and wellness assistant for ConnectFit.\n"
        f"User: {message}\n"
        "Assistant:"
    )
    return generate_text(prompt)
