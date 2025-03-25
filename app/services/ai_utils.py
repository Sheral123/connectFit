import google.generativeai as genai
from app.config import GEMINI_API_KEY, MODEL_NAME
from langchain.llms.base import LLM

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class GeminiLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        response = genai.GenerativeModel(MODEL_NAME).generate_content(prompt)
        return response.text.strip()
