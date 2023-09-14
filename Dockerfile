# base image
FROM python:3.9.6

# 작업 경로
WORKDIR /sound-leader

COPY requirements.txt ./requirements.txt

# PIP upgrade
RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt

# 소스파일 모두 복사
COPY . .

# 포트 열기
EXPOSE 8000

# 실행 : 작동코드
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]