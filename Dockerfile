FROM python:3.10-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사해서 캐시 활용
COPY requirement.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt

# 앱 코드 복사
COPY . .

# Streamlit 포트
EXPOSE 8501

# Streamlit 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]