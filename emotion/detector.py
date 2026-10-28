
from pathlib import Path
from transformers import pipeline

# Local model directory
LOCAL_MODEL_PATH = Path(__file__).resolve().parent / "model"

# Uploaded Hugging Face model repository
HF_MODEL_ID = "LahiruLuci/innerglow-emotion-model"

# Prefer local model files when available
if (LOCAL_MODEL_PATH / "model.safetensors").is_file():
    MODEL_SOURCE = str(LOCAL_MODEL_PATH)
    print("Loading InnerGlow emotion model from local files...")
else:
    MODEL_SOURCE = HF_MODEL_ID
    print("Loading InnerGlow emotion model from Hugging Face...")

classifier = pipeline(
    "text-classification",
    model=MODEL_SOURCE,
    tokenizer=MODEL_SOURCE,
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
