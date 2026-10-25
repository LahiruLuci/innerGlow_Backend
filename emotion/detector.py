from pathlib import Path
from transformers import pipeline

MODEL_PATH = Path(__file__).resolve().parent / "model"

classifier = pipeline(
    "text-classification",
    model=str(MODEL_PATH),
    tokenizer=str(MODEL_PATH),
    device=-1
)

def predict_emotion(text: str):
    result = classifier(
        text,
        truncation=True,
        max_length=128
    )[0]

    return {
        "emotion": result["label"],
        "confidence": round(result["score"], 4)
    }