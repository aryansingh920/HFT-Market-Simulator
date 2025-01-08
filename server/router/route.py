from fastapi import APIRouter
from controllers.root_controller import get_root_message

router = APIRouter()


@router.get("/")
async def root():
    return get_root_message()
