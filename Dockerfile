FROM python:3.10-slim

# ===========================
# Dependencias del sistema
# ===========================
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# ===========================
# Directorio principal
# ===========================
WORKDIR /app

# ===========================
# Github
# ===========================
RUN git clone \
    -b w-okada \
    https://github.com/PESZzzz/Estaagonianuncaacabara.git \
    /opt/voice-changer

# ===========================
# Instalar dependencias oficiales
# ===========================
WORKDIR /opt/voice-changer/server

RUN pip install aiohttp

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# Dependencias del bridge
# ===========================
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    python-multipart

# ===========================
# Volver al proyecto
# ===========================

RUN pip uninstall -y onnxruntime-gpu || true

RUN pip install --no-cache-dir \
    onnxruntime==1.17.3

RUN python3 -c "import onnxruntime; print(onnxruntime.__version__)"

COPY MMVC_Namespace.py /opt/voice-changer/server/sio/MMVC_Namespace.py

EXPOSE 7860

CMD ["python3", "app.py"]
