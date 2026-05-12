import logging

from fastapi import APIRouter

from backend.app.api.responses import success_response

router = APIRouter(tags=["evaluation"])
logger = logging.getLogger(__name__)


@router.post("/evaluation")
def evaluate():
    logger.info("Evaluation endpoint requested")
    return success_response({"status": "pending"})
