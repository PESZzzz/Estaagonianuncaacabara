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
    print("[SocketIO] Conectado al backend")

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
    print("Namespaces:", sio.namespaces)

# ============================================================
# CONFIGURACIÓN (Limpia de /opt/ y comprobaciones de Python antiguas)
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_CHANGER_DIR = os.path.join(APP_DIR, "voice-changer")
# Apuntamos directamente a donde se extraiga tu launcher en dist/main
SERVER_DIR = os.path.join(VOICE_CHANGER_DIR, "dist", "main") 
BACKEND_PORT = 8000
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"

MODEL_PATH = os.path.join(APP_DIR, "pyra.pth")
INDEX_PATH = os.path.join(APP_DIR, "pyra.index")
SLOT_PYRA = 5  

print("=" * 60)
print("Pyra Bridge - HF Edition")
print("=" * 60)

# Comprobamos que exista la carpeta del launcher compilado
if not os.path.isdir(SERVER_DIR):
    raise RuntimeError(f"No existe la carpeta del launcher en: {SERVER_DIR}")

# Determinamos la ruta del ejecutable compilado (el binario de Linux suele llamarse "main")
LAUNCHER_EXE = os.path.join(SERVER_DIR, "main")
if not os.path.isfile(LAUNCHER_EXE):
    # Si no se llama "main", intentamos buscar un ejecutable sin extensión en esa carpeta
    raise RuntimeError(f"No se encontró el ejecutable principal en {SERVER_DIR}")
# =============================================================

PRETRAIN = os.path.join(SERVER_DIR, "pretrain")

print("========== PRETRAIN ==========")
if os.path.exists(PRETRAIN):
    for root, dirs, files in os.walk(PRETRAIN):
        for f in files:
            print(os.path.join(root, f))
print("==============================")

# ============================================================
# ARRANCAR BACKEND (Ejecutando el binario compilado directo)
# ============================================================
# Le damos permisos de ejecución al binario por si acaso
os.chmod(LAUNCHER_EXE, 0o755)

command = [
    LAUNCHER_EXE,
    "-p", str(BACKEND_PORT),
    "--host", "127.0.0.1",
    "--https", "False"
]

backend = subprocess.Popen(command, cwd=SERVER_DIR)

# ============================================================
# LIMPIEZA
# ============================================================
def stop_backend():
    global backend
    if backend is None: return
    if backend.poll() is not None: return
    print("\n[Cleanup] Cerrando backend...")
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
# ESPERAR BACKEND
# ============================================================
print("\nEsperando backend (y descargas iniciales de Okada)...\n")
backend_ok = False
for i in range(600):  
    if backend.poll() is not None:
        raise RuntimeError(f"El launcher compilado terminó prematuramente. ExitCode={backend.returncode}")
    try:
        r = requests.get(BACKEND_URL, timeout=1)
        backend_ok = True
        break
    except Exception:
        if i % 10 == 0:
            print(f"[{i}s] Esperando a que el backend termine de descargar/iniciar...")
        time.sleep(1)

if not backend_ok:
    stop_backend()
    raise RuntimeError("El backend nunca abrió el puerto 8000 tras 10 minutos.")

print("\nBackend iniciado correctamente.")

# ============================================================
# FASTAPI & ROUTING
# ============================================================
app = FastAPI()

@app.on_event("startup")
async def startup():
    await connect_backend()

http = requests.Session()

# ============================================================
# COPIAR Y CARGAR MODELO
# ============================================================
UPLOAD_DIR = os.path.join(SERVER_DIR, "upload_dir")
os.makedirs(UPLOAD_DIR, exist_ok=True)

shutil.copy2(MODEL_PATH, os.path.join(UPLOAD_DIR, "pyra.pth"))
shutil.copy2(INDEX_PATH, os.path.join(UPLOAD_DIR, "pyra.index"))

def cargar_pyra():
    print("\n[Bridge RVC] Solicitando carga de Pyra en Slot 5...")
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
        print(f"[Backend Okada] Carga completada. Status: {r.status_code}")
    except Exception as e:
        print(f"❌ Error al inyectar modelo: {e}")

print("Esperando 60 segundos antes de cargar Pyra...")
time.sleep(60)
cargar_pyra()

# ============================================================
# ENDPOINTS DE INFERENCIA & INTERFAZ
# ============================================================
VC_ENDPOINTS = ["/rest/vc", "/rest/vc_stream", "/rest/receive", "/infer", "/api/v1/infer"]

import traceback
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

        print("Enviando SocketIO...")

        await sio.emit(
            "request_message",
            [timestamp, pcm.tobytes()],
            namespace="/test"
        )

        print("Esperando respuesta...")

        audio = await asyncio.wait_for(future, timeout=5)

        print("Respuesta recibida:", len(audio))

        return Response(
            content=audio,
            media_type="application/octet-stream"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(str(e), status_code=500)

# LA RUTA RAÍZ QUE SALVA EL SPACE DE HF Y GESTIONA EL AUDIO EN TIEMPO REAL
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>RVC Live Voice Changer Studio - Pyra Edition</title>
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
        <div class="subtitle">Pyra Bridge — Slot 5</div>

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
        let activeOutput;
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

    if (processing) {
        return;
    }

    processing = true;

    console.log("🎤 Nuevo chunk");
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

console.log("✅ Backend respondió");
                        if (response.ok && response.status === 200) {
                            const audioBlob = await response.blob();
                            const audioUrl = URL.createObjectURL(audioBlob);
                            
                            const chunkAudio = new Audio(audioUrl);
                            chunkAudio.play().catch(() => {});
                        } else {
                            console.error("Fallo en inferencia backend:", response.status);
                        }
                    } catch (err) {

    console.error("🔥 Error:", err);

} finally {

    processing = false;

}
                };

                source.connect(processor);
                processor.connect(audioContext.destination);

            } catch (error) {

    console.error("========== ERROR ==========");
    console.error(error);
    console.error("Nombre:", error.name);
    console.error("Mensaje:", error.message);
    console.error("===========================");

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

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
