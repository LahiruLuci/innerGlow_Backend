
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from emotion.detector import predict_emotion
from chatbot import generate_response


# Create FastAPI application
app = FastAPI(
    title="InnerGlow Emotion Detection API",
    version="1.0.0"
)


# Allow Next.js frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request structure for emotion detection
class EmotionRequest(BaseModel):
    text: str


# Request structure for chatbot
class ChatRequest(BaseModel):
    message: str


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


# Chatbot endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    # Detect emotion
    prediction = predict_emotion(request.message)

    # Generate chatbot response
    reply = generate_response(
        message=request.message,
        emotion=prediction["emotion"]
    )

    # Return response to frontend
    return {
        "message": request.message,
        "emotion": prediction["emotion"],
        "confidence": prediction["confidence"],
        "reply": reply
    }
