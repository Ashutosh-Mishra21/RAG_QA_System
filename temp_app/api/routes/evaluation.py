from fastapi import APIRouter

router = APIRouter(tags=["evaluation"])


@router.post("/evaluation")
def evaluate():
    return {"status": "pending"}
