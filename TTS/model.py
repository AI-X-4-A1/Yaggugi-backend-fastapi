from pydantic import BaseModel


# 텍스트 입력 모델 정의
class TextInput(BaseModel):
    text: str
