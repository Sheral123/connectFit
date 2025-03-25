from fastapi import FastAPI
from app.api import genai

app = FastAPI(title="ConnectFit GenAI API")

# Mount the GenAI endpoints under /api/genai
app.include_router(genai.router, prefix="/api/genai")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
