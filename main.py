from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from gtts import gTTS
from io import BytesIO
from transformers import pipeline
from groq import Groq
from contextlib import asynccontextmanager

from OCR.paddleocr.paddleocr import PaddleOCR
from OCR.paddleocr.utils import *
from TTS.cors_config import setup_cors  # 외부 CORS 설정 모듈 가져오기
from TTS.model import TextInput

import os
import json  # JSON 파싱을 위한 모듈 추가
import re  # 날짜 형식 검사 및 수정용

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# fastapi 시작 전 초기화 작업
def start():
    print("Yaggugi-backend-fastapi start")
    # OCR 모델 초기화
    app.state.ocr_en_model = PaddleOCR(lang="en", use_gpu=True)
    app.state.ocr_kr_model = PaddleOCR(lang="korean", use_gpu=True)

    # Load the Whisper model for ASR
    app.state.transcriber = pipeline(
        model="openai/whisper-large", task="automatic-speech-recognition"
    )

    # Set up Groq client
    app.state.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")


# fastapi 종료 전 마무리 작업
def shutdown():
    print("Yaggugi-backend-fastapi stop")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start()
    yield
    shutdown()


app = FastAPI(lifespan=lifespan)  # FastAPI 앱 초기화

setup_cors(app)

@app.post("/ocr")
async def create_upload_files(file: UploadFile):
    content = await file.read()

    ocr_texts = []  # 순서 유지
    ocr_unique_texts = set()  # 중복 체크
    # preprocess
    content = prepreprocess_text(content)

    # OCR 모델 사용
    ocr_en_model = app.state.ocr_en_model
    ocr_kr_model = app.state.ocr_kr_model
    get_ocr_result(content, ocr_texts, ocr_unique_texts, ocr_en_model)  # en
    get_ocr_result(content, ocr_texts, ocr_unique_texts, ocr_kr_model)  # ko

    # postpreprocess
    ocr_pp_unique_text = postpreprocess_text(ocr_texts, ocr_unique_texts)
    print(f"ocr_pp_unique_text: {ocr_pp_unique_text}")

    return {"ocr_result": ocr_pp_unique_text}


# TTS 변환 엔드포인트 정의
@app.post("/synthesize")
async def synthesize_text(input: TextInput):
    # 입력된 텍스트가 비어 있는지 확인
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="텍스트를 입력해야 합니다.")

    # gTTS 객체 생성 및 텍스트 변환
    tts = gTTS(text=input.text, lang="ko")

    # 바이트 스트림으로 음성 데이터를 저장
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    # 음성 데이터를 스트리밍하여 응답
    return StreamingResponse(audio_bytes, media_type="audio/mpeg")


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    # Save the uploaded file
    file_location = f"temp/{file.filename}"
    os.makedirs(
        os.path.dirname(file_location), exist_ok=True
    )  # Create temp directory if it doesn't exist
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Transcribe audio to text
    transcriber = app.state.transcriber
    result = transcriber(file_location)
    text = result["text"]

    # Delete the temporary file
    os.remove(file_location)

    # Request sentiment analysis and date extraction from Groq API
    client = app.state.client
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": f"""
                Please analyze the following text to determine if the speaker took medicine or not. 
                If the speaker took medicine, respond with "positive"; if not, respond with "negative". Also, identify any dates mentioned 
                in the text. Respond strictly in the following JSON format, without additional commentary:
            
                {{
                "sentiment": "positive" or "negative",
                "mentioned_date": "MM-DD" (if any date is mentioned, otherwise null)
                }}
            
                Here is the text: "{text}"
                """
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    # Generate result JSON from Groq completion
    result_json  = ""
    for chunk in completion:
        result_json += chunk.choices[0].delta.content or ""

    # Parse the result JSON and add default year to the date
    try:
        parsed_result = json.loads(result_json)  # JSON 문자열을 파싱
        # 날짜가 MM-DD 형식이라면 2024년을 추가
        if parsed_result.get("mentioned_date"):
            if re.match(r"^\d{2}-\d{2}$", parsed_result["mentioned_date"]):
                parsed_result["mentioned_date"] = f"2024-{parsed_result['mentioned_date']}"
        print("Parsed Sentiment Analysis Result:", parsed_result)  # 터미널에 JSON 출력

    except json.JSONDecodeError:
        print("Failed to parse JSON response:", result_json)  # JSON 파싱 실패 시 출력

    # Log the results for debugging
    print("Transcribed Text:", text)
    print("Sentiment Analysis Result (Raw JSON):", result_json)

    return {
        "transcribed_text": text, 
        "sentiment_analysis_result": parsed_result if 'parsed_result' in locals() else result_json
    }
