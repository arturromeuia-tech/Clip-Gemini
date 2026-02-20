
import os, uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import FileResponse, HTMLResponse
from pipeline import process_video

app = FastAPI()

# ── HTML embebido ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>🎬 Video Clip Cropper</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: #0f0f0f;
      color: #f0f0f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .card {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 16px;
      padding: 40px;
      width: 100%;
      max-width: 600px;
    }
    h1 { font-size: 1.6rem; margin-bottom: 8px; }
    p.sub { color: #888; font-size: 0.9rem; margin-bottom: 32px; }
    .field { margin-bottom: 20px; }
    label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 6px; }
    input[type="password"], input[type="file"] {
      width: 100%;
      padding: 12px 16px;
      background: #111;
      border: 1px solid #333;
      border-radius: 10px;
      color: #fff;
      font-size: 0.95rem;
      outline: none;
    }
    input[type="file"] { cursor: pointer; }
    button {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      margin-top: 8px;
      transition: opacity 0.2s;
    }
    button:hover { opacity: 0.9; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .status {
      margin-top: 24px;
      padding: 16px;
      background: #111;
      border-radius: 10px;
      font-size: 0.88rem;
      color: #ccc;
      white-space: pre-line;
      display: none;
      line-height: 1.7;
    }
    .status.show  { display: block; }
    .status.error   { color: #f87171; border: 1px solid #7f1d1d; }
    .status.success { color: #86efac; border: 1px solid #14532d; }
    .download-btn {
      display: none;
      margin-top: 16px;
      text-align: center;
      padding: 12px;
      background: #16a34a;
      color: white;
      border-radius: 10px;
      font-weight: 600;
      text-decoration: none;
      transition: opacity 0.2s;
    }
    .download-btn:hover { opacity: 0.85; }
    .download-btn.show { display: block; }
    .loader {
      display: inline-block;
      width: 14px; height: 14px;
      border: 2px solid #fff;
      border-top-color: transparent;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin-right: 8px;
      vertical-align: middle;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎬 Video Clip Cropper</h1>
    <p class="sub">Detecta y recorta el clip incrustado en tu diseño con Gemini + SAM2</p>

    <div class="field">
      <label>🔑 Gemini API Key</label>
      <input type="password" id="apiKey" placeholder="AIza..." />
    </div>

    <div class="field">
      <label>📤 Vídeo MP4</label>
      <input type="file" id="videoFile" accept=".mp4" />
    </div>

    <button id="btn" onclick="processVideo()">🚀 Detectar y Recortar</button>

    <div class="status" id="status"></div>
    <a class="download-btn" id="downloadBtn" download="clip_recortado.mp4">
      ⬇️ Descargar Clip Recortado
    </a>
  </div>

  <script>
    async function processVideo() {
      const apiKey    = document.getElementById("apiKey").value.trim();
      const fileInput = document.getElementById("videoFile");
      const btn       = document.getElementById("btn");
      const dlBtn     = document.getElementById("downloadBtn");

      if (!apiKey)            return showStatus("❌ Introduce tu Gemini API Key", "error");
      if (!fileInput.files[0]) return showStatus("❌ Selecciona un vídeo .mp4", "error");

      btn.disabled = true;
      btn.innerHTML = '<span class="loader"></span> Procesando...';
      dlBtn.classList.remove("show");
      showStatus("⏳ Subiendo vídeo y procesando...\\nEsto puede tardar unos minutos.", "");

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      try {
        const res = await fetch("/crop", {
          method: "POST",
          headers: { "x-gemini-key": apiKey },
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Error desconocido");
        }

        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);
        const info = res.headers.get("X-Info") || "";

        showStatus("✅ ¡Completado!\\n\\n" + info, "success");
        dlBtn.href = url;
        dlBtn.classList.add("show");

      } catch (e) {
        showStatus("❌ Error: " + e.message, "error");
      } finally {
        btn.disabled  = false;
        btn.innerHTML = "🚀 Detectar y Recortar";
      }
    }

    function showStatus(msg, type) {
      const el = document.getElementById("status");
      el.textContent = msg;
      el.className   = "status show " + type;
    }
  </script>
</body>
</html>
"""

# ── Rutas ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/crop")
async def crop(
    file: UploadFile = File(...),
    x_gemini_key: str = Header(None),
):
    key = x_gemini_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise HTTPException(400, "Gemini API Key requerida")
    os.environ["GEMINI_API_KEY"] = key

    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "Solo se aceptan archivos .mp4")

    content   = await file.read()
    tmp_input = f"/tmp/{uuid.uuid4()}.mp4"

    with open(tmp_input, "wb") as f:
        f.write(content)

    try:
        output_path, info = process_video(tmp_input)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_input):
            os.remove(tmp_input)

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename="clip_recortado.mp4",
        headers={"X-Info": str(info)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
