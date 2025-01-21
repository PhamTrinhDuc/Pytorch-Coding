from fastapi import APIRouter
from .catdog_route import router as catdog_cls_route

router = APIRouter()
router.include_router(catdog_cls_route, 
                      prefix="/catdog_classification")
