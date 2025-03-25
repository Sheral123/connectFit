from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import chat_service, rag_service, nutrition_service

router = APIRouter()

# -------------------------------
# AI Chatbot Endpoint
# -------------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = chat_service.get_chat_response(request.message)
        return ChatResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Personalized Plan Generation (RAG) Endpoint
# -------------------------------
class PlanRequest(BaseModel):
    preferences: str  # e.g., {"goal": "weight loss", "diet": "balanced", "activity": "cardio"}

class PlanResponse(BaseModel):
    plan: str

@router.post("/generate-plan", response_model=PlanResponse)
async def generate_plan_endpoint(request: PlanRequest):
    try:
        plan_text = rag_service.generate_personalized_plan(request.preferences)
        return PlanResponse(plan=plan_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# Nutrition Advice Endpoint
# -------------------------------
class NutritionRequest(BaseModel):
    query: str

class NutritionResponse(BaseModel):
    advice: str

@router.post("/nutrition-advice", response_model=NutritionResponse)
async def nutrition_advice_endpoint(request: NutritionRequest):
    try:
        advice = nutrition_service.get_nutrition_advice(request.query)
        return NutritionResponse(advice=advice)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
