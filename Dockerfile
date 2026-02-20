
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Precarga el modelo SAM2 al buildear (evita espera en primer uso)
RUN python -c "from ultralytics import SAM; SAM('sam2.1_b.pt')"

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
