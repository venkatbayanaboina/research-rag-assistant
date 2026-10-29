FROM python:3.12-slim

# poppler-utils: pdf2image rendering; libgl1/libglib: opencv-headless deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# Model weights download on first query (~2 GB) — mount a cache volume:
#   docker run -v glm_cache:/root/.cache/huggingface ...
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false
EXPOSE 8501
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
