# from fastapi import FastAPI
# from contextlib import asynccontextmanager
# from transformers import pipeline
# from OCR.paddleocr.paddleocr import PaddleOCR
# from groq import Groq
# from dotenv import load_dotenv
# from TTS.cors_config import setup_cors

# import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# # fastapi 시작 전 초기화 작업
# def start():
#     print("Yaggugi-backend-fastapi start")
#     # OCR 모델 초기화
#     app.state.ocr_en_model = PaddleOCR(lang="en")
#     app.state.ocr_kr_model = PaddleOCR(lang="korean")

#     # Load the Whisper model for ASR
#     app.state.transcriber = pipeline(
#         model="openai/whisper-large", task="automatic-speech-recognition"
#     )

#     # Set up Groq client
#     app.state.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#     # CORS 설정 적용
#     setup_cors(app)


# # fastapi 종료 전 마무리 작업
# def shutdown():
#     print("Yaggugi-backend-fastapi stop")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     start()
#     yield
#     shutdown()
