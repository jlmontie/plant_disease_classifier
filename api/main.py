from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from api.inference import load_model, predict_image
import io

app = FastAPI()

model = load_model()

@app.get("/")
def root():
    return {"message": "Plant Disease Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only JPG and PNG files are supported.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = predict_image(model, image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))