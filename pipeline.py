
import os, json, shutil, tempfile
import numpy as np
import cv2
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig, Part
from ultralytics import SAM

SAM_MODEL_PATH = os.getenv("SAM_MODEL", "sam2.1_b.pt")


def detect_bbox_gemini(frame_path: str) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    img    = Image.open(frame_path)
    W, H   = img.size

    with open(frame_path, "rb") as f:
        image_bytes = f.read()

    prompt = """
    Analiza este frame. Tiene un diseño gráfico con fondo decorativo
    y una ventana/panel central donde hay un clip de vídeo incrustado.
    Detecta exactamente los límites de esa ventana y devuelve SOLO JSON:
    {"y1": int, "x1": int, "y2": int, "x2": int}
    Valores normalizados de 0 a 1000. Solo el JSON, nada más.
    """

    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), prompt],
        config=GenerateContentConfig(response_mime_type="application/json", temperature=0.1),
    )

    b = json.loads(res.text)
    pad = 4
    return {
        "x1": max(0, int(b["x1"] / 1000 * W) - pad),
        "y1": max(0, int(b["y1"] / 1000 * H) - pad),
        "x2": min(W, int(b["x2"] / 1000 * W) + pad),
        "y2": min(H, int(b["y2"] / 1000 * H) + pad),
        "W": W, "H": H,
    }


def refine_with_sam(frame_path: str, bbox: dict) -> dict:
    try:
        sam     = SAM(SAM_MODEL_PATH)
        results = sam(frame_path, bboxes=[bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]])
        masks   = results[0].masks
        if masks is None or len(masks.data) == 0:
            return bbox
        mask = masks.data[0].cpu().numpy().astype(np.uint8)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return bbox
        return {
            "x1": int(xs.min()), "y1": int(ys.min()),
            "x2": int(xs.max()), "y2": int(ys.max()),
            "W": bbox["W"],      "H": bbox["H"],
        }
    except Exception:
        return bbox


def process_video(video_path: str) -> tuple:
    tmpdir     = tempfile.mkdtemp()
    frames_dir = f"{tmpdir}/frames"
    crop_dir   = f"{tmpdir}/cropped"
    os.makedirs(frames_dir)
    os.makedirs(crop_dir)

    try:
        # 1 — Extraer frames
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        idx   = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{frames_dir}/{idx:06d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            idx += 1
        cap.release()
        total_frames = idx

        # 2 — Gemini en frame 0
        first_frame = f"{frames_dir}/000000.jpg"
        bbox_gemini = detect_bbox_gemini(first_frame)

        # 3 — SAM2 refina
        bbox_final  = refine_with_sam(first_frame, bbox_gemini)
        x1, y1      = bbox_final["x1"], bbox_final["y1"]
        x2, y2      = bbox_final["x2"], bbox_final["y2"]
        crop_w      = x2 - x1
        crop_h      = y2 - y1

        # 4 — Recortar todos los frames
        for fname in sorted(os.listdir(frames_dir)):
            frame = cv2.imread(f"{frames_dir}/{fname}")
            crop  = frame[y1:y2, x1:x2]
            cv2.imwrite(f"{crop_dir}/{fname}", crop)

        # 5 — Reconstruir MP4
        output_path = f"{tmpdir}/output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))
        for fname in sorted(os.listdir(crop_dir)):
            frame = cv2.imread(f"{crop_dir}/{fname}")
            if frame is not None:
                out.write(frame)
        out.release()

        final_path = "/tmp/output_cropped.mp4"
        shutil.copy(output_path, final_path)

        return final_path, {
            "frames": total_frames,
            "fps": round(fps, 2),
            "bbox_gemini": bbox_gemini,
            "bbox_sam2":   bbox_final,
            "resolucion":  f"{crop_w}x{crop_h}",
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
