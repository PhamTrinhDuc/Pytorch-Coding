import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile
from fastapi import APIRouter
from schemas.catdog_schema import CatDogResponse
from model.catdot_predictor import CatDogPredictor
from config import ModelConfig

config = ModelConfig()

predictor = CatDogPredictor(
    model_name=config.MODEL_NAME,
    model_path=config.MODEL_WEIGHT,
    device=config.DEVICE
)

router = APIRouter()

@router.post("/predict", response_model=CatDogResponse)
async def predict(file_upload: UploadFile = File(...)):

    response = await predictor.predict(
        image=file_upload.file
    )
    
    return response
