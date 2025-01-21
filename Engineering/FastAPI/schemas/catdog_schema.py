from pydantic import BaseModel

class CatDogResponse(BaseModel):
    probs: list = []
    best_probs: float = -1.0
    predicted_class: str = ""
    predictor_name: str = ""
