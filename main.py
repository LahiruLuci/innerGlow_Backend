from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from emotion.detector import predict_emotion
from chatbot import generate_response

# Create FastAPI application
app = FastAPI(
    title="InnerGlow Emotion Detection API",
    version="1.0.0"
)

# Request structure
class EmotionRequest(BaseModel):
    text: str

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "InnerGlow Backend is running!"
    }

# Emotion detection endpoint
@app.post("/emotion")
def detect_emotion(request: EmotionRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    return predict_emotion(request.text)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    prediction = predict_emotion(request.message)

    reply = generate_response(
        message=request.message,
        emotion=prediction["emotion"]
    )

    return {
        "message": request.message,
        "emotion": prediction["emotion"],
        "confidence": prediction["confidence"],
        "reply": reply
    }
