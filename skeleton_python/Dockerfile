FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* 
    # Clean up apt cache to reduce image size

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess/ ./preprocess/
COPY skeletonize/ ./skeletonize/
COPY pipeline/ ./pipeline/

CMD ["python", "--version"]
