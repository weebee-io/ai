# controller/users.py

from typing import Union
from fastapi import APIRouter

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/{user_id}")
def read_item(user_id: int):
    return {"user_id": user_id}