FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 설치
RUN apt-get update && apt-get install --no-install-recommends -y \
    wget \
    pkg-config \
    python3-dev \
    default-libmysqlclient-dev \
    build-essential \
    libegl1 \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    python3-pip \
    ffmpeg && \
    apt-get clean

# CUDA 설치
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit && \
    rm cuda_11.8.0_520.61.05_linux.run

# 환경 변수 설정
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# requirements.txt 복사 및 의존성 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 파일 복사
COPY . .

# 환경 변수 로드용 .env 파일 복사
COPY .env .env

# numpy 2.x 에러 방지
COPY ./OCR/imgaug.py /usr/local/lib/python3.12/site-packages/imgaug/imgaug.py

EXPOSE 8090

# 컨테이너 시작 시 실행할 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]