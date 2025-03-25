import os
import google.generativeai as genai
from app.config import GEMINI_API_KEY, MODEL_NAME

# Configure the Gemini API with your key.
genai.configure(api_key='AIzaSyDxwytto_T5C1HVOzO119pnK0euPPqPe80')

# Initialize a reference to the Gemini model.
# Depending on your plan, you might use a different model identifier.
# For example, "gemini-pro" is used here.
def generate_text(prompt: str) -> str:
    try:
        # Generate content using the Gemini model.
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"
    


