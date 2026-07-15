import itertools
import socketio
import struct
import numpy as np
import asyncio
import os
import sys
import time
import signal
import atexit
import subprocess
import shutil
import json
import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse

sio = socketio.AsyncClient()

backend_ready = False
pending = {}
counter = itertools.count()

@sio.event
async def connect():
    global backend_ready
    backend_ready = True
    print("[SocketIO] Conectado al backend de Okada")

@sio.on("response", namespace="/test")
async def on_response(msg):
    try:
        timestamp = msg[0]
        audio = msg[1]
        future = pending.pop(timestamp, None)
        if future:
            future.set_result(audio)
    except Exception as e:
        print("Error en respuesta SocketIO:", e)

async def connect_backend():
    global backend_ready
    if backend_ready:
        return
    print("Conectando SocketIO...")
    await sio.connect(
        f"http://127.0.0.1:{BACKEND_PORT}",
        namespaces=["/test"]
    )
    print("SocketIO conectado:", sio.connected)

# ============================================================
# CONFIGURACIÓN (Rutas dinámicas basadas en código de Colab)
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_CHANGER_DIR = os.path.join(APP_DIR, "voice-changer")
SERVER_DIR = os.path.join(VOICE_CHANGER_DIR, "server")
BACKEND_PORT = 8000
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

# Rutas de tus modelos
MODEL_PATH = os.path.join(APP_DIR, "modelos", "pyra", "pyra.pth")
INDEX_PATH = os.path.join(APP_DIR, "modelos", "pyra", "pyra.index")
SLOT_PYRA = 5  

print("=" * 60)
print("Pyra Bridge - Colab Base Edition")
print("=" * 60)

# Verificamos que el script de Okada exista
SERVER_FILE = os.path.join(SERVER_DIR, "MMVCServerSIO.py")
if not os.path.isfile(SERVER_FILE):
    raise RuntimeError(f"No se encontró el script de Okada en {SERVER_FILE}")

# ============================================================
# INICIAR BACKEND (Usa exactamente los mismos argumentos del Colab)
# ============================================================
command = [
    "python3", "MMVCServerSIO.py",
    "-p", str(BACKEND_PORT),
    "--https", "False",
    "--content_vec_500", "pretrain/checkpoint_best_legacy_500.pt",
    "--content_vec_500_onnx", "pretrain/content_vec_500.onnx",
    "--content_vec_500_onnx_on", "true",
    "--hubert_base", "pretrain/hubert_base.pt",
    "--hubert_base_jp", "pretrain/rinna_hubert_base_jp.pt",
    "--hubert_soft", "pretrain/hubert/hubert-soft-0d54a1f4.pt",
    "--nsf_hifigan", "pretrain/nsf_hifigan/model",
    "--crepe_onnx_full", "pretrain/crepe_onnx_full.onnx",
    "--crepe_onnx_tiny", "pretrain/crepe_onnx_tiny.onnx",
    "--rmvpe", "pretrain/rmvpe.pt",
    "--model_dir", "model_dir",
    "--samples", "samples.json"
]

backend = subprocess.Popen(command, cwd=SERVER_DIR)

# ============================================================
# GESTIÓN DE APAGADO
# ============================================================
def stop_backend():
    global backend
    if backend is None: return
    if backend.poll() is not None: return
    print("\n[Cleanup] Apagando servidor de Okada...")
    try:
        backend.terminate()
        backend.wait(timeout=5)
    except Exception:
        try: backend.kill()
        except Exception: pass

atexit.register(stop_backend)
def shutdown_handler(sig, frame):
    stop_backend()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# ============================================================
# ESPERAR INICIO Y DESCARGAS DE OKADA
# ============================================================
# El primer arranque tardará un poco porque descargará los .pt y .onnx de pretrain
print("\nEsperando que Okada inicie (y descargue los modelos pretrain en segundo plano)...\n")
backend_ok = False
for i in range(900):  # 15 minutos de límite por si las descargas van lento
    if backend.poll() is not None:
        raise RuntimeError(f"El backend de Okada falló al iniciar. Código de salida={backend.returncode}")
    try:
        r = requests.get(BACKEND_URL, timeout=1)
        backend_ok = True
        break
    except Exception:
        if i % 15 == 0:
            print(f"[{i}s] Esperando servidor local (Okada está descargando dependencias)...")
        time.sleep(1)

if not backend_ok:
    stop_backend()
    raise RuntimeError("El backend de Okada no levantó después de 15 minutos.")

print("\nBackend de Okada iniciado de forma exitosa en Linux.")

# ============================================================
# FASTAPI & INYECCIÓN DEL MODELO PYRA
# ============================================================
app = FastAPI()

@app.on_event("startup")
async def startup():
    await connect_backend()

http = requests.Session()

# Copiamos Pyra dentro de la carpeta upload_dir de Okada
UPLOAD_DIR = os.path.join(SERVER_DIR, "upload_dir")
os.makedirs(UPLOAD_DIR, exist_ok=True)
shutil.copy2(MODEL_PATH, os.path.join(UPLOAD_DIR, "pyra.pth"))
shutil.copy2(INDEX_PATH, os.path.join(UPLOAD_DIR, "pyra.index"))

def cargar_pyra():
    print("\n[Bridge RVC] Cargando modelo Pyra en el Slot 5...")
    params = {
        "voiceChangerType": "RVC",
        "slot": SLOT_PYRA,
        "isSampleMode": False,
        "sampleId": "",
        "files": [
            {"name": "pyra.pth", "kind": "rvcModel", "dir": ""},
            {"name": "pyra.index", "kind": "rvcIndex", "dir": ""}
        ],
        "params": {
            "slot": SLOT_PYRA,
            "name": "Pyra",
            "modelType": "pyTorchRVC",
            "version": "v2",
            "samplingRate": 40000,
            "f0": True,
            "embChannels": 768,
            "embedder": "hubert_base",
            "useFinalProj": False,
            "defaultTune": 0,
            "defaultIndexRatio": 0.7,
            "defaultProtect": 0.5
        }
    }
    
    try:
        r = http.post(
            BACKEND_URL + "/load_model",
            data={"slot": SLOT_PYRA, "isHalf": "false", "params": json.dumps(params)},
            timeout=30
        )
        print(f"[Backend Okada] Inyección de modelo exitosa. Código: {r.status_code}")
    except Exception as e:
        print(f"❌ Error al inyectar modelo: {e}")

# Pausa para que se estabilice el servidor de Okada antes de inyectar el slot
print("Esperando 20 segundos antes de inyectar Pyra...")
time.sleep(20)
cargar_pyra()

# ============================================================
# ENDPOINT DE AUDIO E INTERFAZ WEB
# ============================================================
@app.post("/api/stream_bridge")
async def stream_bridge(request: Request):
    try:
        form = await request.form()
        wav = await form["audio"].read()
        pcm = np.frombuffer(wav[44:], dtype=np.int16)
        timestamp = next(counter)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        pending[timestamp] = future

        await sio.emit(
            "request_message",
            [timestamp, pcm.tobytes()],
            namespace="/test"
        )

        audio = await asyncio.wait_for(future, timeout=5)
        return Response(content=audio, media_type="application/octet-stream")
    except Exception as e:
        return Response(str(e), status_code=500)

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>RVC Live Voice Changer - Pyra Colab Edition</title>
    <style>
        body { font-family: system-ui, sans-serif; background-color: #0f172a; color: #f8fafc; margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
        .container { background: #1e293b; padding: 2.5rem; border-radius: 16px; width: 100%; max-width: 500px; text-align: center; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
        h1 { color: #f43f5e; margin-bottom: 0.5rem; }
        .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
        .slider-container { display: flex; flex-direction: column; text-align: left; margin-bottom: 1.5rem; gap: 0.5rem; }
        .slider-header { display: flex; justify-content: space-between; font-weight: 500; }
        .val-badge { background: #f43f5e; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.85rem; }
        input[type="range"] { width: 100%; accent-color: #f43f5e; cursor: pointer; }
        #status { font-weight: bold; font-size: 1.2rem; margin: 2rem 0; color: #64748b; padding: 0.5rem; border-radius: 8px; background: #0f172a; }
        .btn { width: 100%; padding: 0.85rem; border: none; border-radius: 8px; font-weight: bold; font-size: 1rem; cursor: pointer; transition: all 0.2s; color: white; margin-bottom: 0.7rem; }
        .btn-start { background-color: #10b981; }
        .btn-start:hover { background-color: #059669; }
        .btn-stop { background-color: #ef4444; }
        .btn-stop:hover { background-color: #dc2626; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 RVC Voice Changer</h1>
        <div class="subtitle">Pyra Bridge (Basado en Colab Linux)</div>

        <div class="slider-container">
            <div class="slider-header">
                <label for="pitch">Pitch (Tono):</label>
                <span id="pitch_val" class="val-badge">0</span>
            </div>
            <input type="range" id="pitch" min="-12" max="12" value="0" oninput="document.getElementById('pitch_val').innerText=this.value">
        </div>

        <div class="slider-container">
            <div class="slider-header">
                <label for="index">Index Rate:</label>
                <span id="index_val" class="val-badge">0.7</span>
            </div>
            <input type="range" id="index" min="0" max="1" value="0.7" step="0.05" oninput="document.getElementById('index_val').innerText=this.value">
        </div>

        <div id="status">⚪ DETENIDO</div>

        <button class="btn btn-start" id="btn_start" onclick="startLiveVoice()">🎙️ Iniciar Conversión</button>
        <button class="btn btn-stop" id="btn_stop" style="display:none;" onclick="stopLiveVoice()">🛑 Detener</button>
    </div>

    <script>
        let audioContext;
        let mediaStream;
        let processor;
        let processing = false;

        function bufferToWav(buffer) {
            let length = buffer.length * 2;
            let bufferArr = new ArrayBuffer(44 + length);
            let view = new DataView(bufferArr);

            function writeString(offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            writeString(0, 'RIFF');
            view.setUint32(4, 36 + length, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, 44100, true);
            view.setUint32(28, 44100 * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, length, true);

            let offset = 44;
            for (let i = 0; i < buffer.length; i++, offset += 2) {
                let s = Math.max(-1, Math.min(1, buffer[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }
            return new Blob([view], { type: 'audio/wav' });
        }

        async function startLiveVoice() {
            const statusLabel = document.getElementById('status');
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(256, 1, 1);

                document.getElementById('btn_start').style.display = 'none';
                document.getElementById('btn_stop').style.display = 'block';
                statusLabel.innerText = "🔊 TRANSMITIENDO AUDIO...";
                statusLabel.style.color = "#10b981";

                processor.onaudioprocess = async function(e) {
                    if (processing) return;
                    processing = true;

                    const inputData = e.inputBuffer.getChannelData(0);
                    const wavBlob = bufferToWav(inputData);

                    const formData = new FormData();
                    formData.append("audio", wavBlob, "audio.wav");
                    formData.append("pitch", document.getElementById('pitch').value);
                    formData.append("index_rate", document.getElementById('index').value);

                    try {
                        const response = await fetch('/api/stream_bridge', {
                            method: 'POST',
                            body: formData
                        });

                        if (response.ok && response.status === 200) {
                            const audioBlob = await response.blob();
                            const audioUrl = URL.createObjectURL(audioBlob);
                            const chunkAudio = new Audio(audioUrl);
                            chunkAudio.play().catch(() => {});
                        }
                    } catch (err) {
                        console.error("Fallo de transmisión:", err);
                    } finally {
                        processing = false;
                    }
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

            } catch (error) {
                statusLabel.innerText = "❌ ERROR: PERMISO DENEGADO";
                statusLabel.style.color = "#ef4444";
            }
        }

        function stopLiveVoice() {
            if (processor) processor.disconnect();
            if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
            if (audioContext) audioContext.close();
            
            document.getElementById('btn_start').style.display = 'block';
            document.getElementById('btn_stop').style.display = 'none';
            
            const statusLabel = document.getElementById('status');
            statusLabel.innerText = "⚪ DETENIDO";
            statusLabel.style.color = "#64748b";
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return HTML_CONTENT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
