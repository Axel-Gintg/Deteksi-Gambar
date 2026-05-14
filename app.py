"""
Flask API Backend — Deteksi Manipulasi Gambar
=============================================
Menyediakan endpoint REST untuk inference model EfficientNet-B4
yang dilatih pada dataset CASIA untuk mendeteksi manipulasi gambar.

Menggunakan Multi-Layer Grad-CAM ensemble untuk lokalisasi yang akurat.
"""

import io
import os
import base64
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM, LayerCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_efficientnetb4_casia_full.pth"
IMG_SIZE   = 380
ELA_QUALITIES = [90, 75, 60]  # Multi-quality ELA
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Setup ─────────────────────────────────────────────────────────────
def load_model():
    """Muat model EfficientNet-B4 yang sudah di-train."""
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)

    state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)

    # Handle berbagai format state_dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    # Cek apakah checkpoint menggunakan custom head (head.1, head.4)
    has_custom_head = any(k.startswith("head.") for k in state.keys())

    if has_custom_head:
        # Checkpoint menggunakan custom head:
        #   head.0 = Flatten/AdaptiveAvgPool (non-parametric)
        #   head.1 = Linear(1792, 512)
        #   head.2 = ReLU (non-parametric)
        #   head.3 = Dropout (non-parametric)
        #   head.4 = Linear(512, 2)
        # Ganti classifier timm default dengan custom head yang sesuai
        model.classifier = nn.Sequential(
            nn.Flatten(1),           # 0
            nn.Linear(1792, 512),    # 1
            nn.ReLU(inplace=True),   # 2
            nn.Dropout(0.3),         # 3
            nn.Linear(512, 2),       # 4
        )

    # Bersihkan prefix 'module.' dan 'backbone.' dari keys
    cleaned = {}
    for k, v in state.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        # Map 'head.X' → 'classifier.X' agar sesuai dengan model timm
        if key.startswith("head."):
            key = "classifier." + key[len("head."):]
        cleaned[key] = v

    # Load state dict
    try:
        model.load_state_dict(cleaned, strict=True)
        print("   ✅ State dict loaded (strict)")
    except RuntimeError as e:
        print(f"   ⚠️ Strict load gagal, mencoba non-strict: {e}")
        model.load_state_dict(cleaned, strict=False)
        print("   ✅ State dict loaded (non-strict)")

    model.to(DEVICE)
    model.eval()
    return model

print("🔄 Memuat model EfficientNet-B4...")
model = load_model()
print(f"✅ Model dimuat di {DEVICE}")

# ─── Transforms ──────────────────────────────────────────────────────────────
val_tfm = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ─── ELA ─────────────────────────────────────────────────────────────────────
def convert_to_ela(image_path: str, quality: int = 90) -> Image.Image:
    """Error Level Analysis: re-save dengan kompresi JPEG lalu hitung diff."""
    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(original, resaved)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela


def convert_to_ela_from_pil(pil_img: Image.Image, quality: int = 90) -> Image.Image:
    """ELA dari PIL Image langsung (tanpa save ke disk)."""
    original = pil_img.convert("RGB")
    # Save sebagai JPEG di memory
    buf_orig = io.BytesIO()
    original.save(buf_orig, "JPEG", quality=95)
    buf_orig.seek(0)
    saved_orig = Image.open(buf_orig).convert("RGB")

    buf = io.BytesIO()
    saved_orig.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    ela = ImageChops.difference(saved_orig, resaved)
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Konversi PIL Image ke base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Multi-Layer Ensemble Grad-CAM ──────────────────────────────────────────
def compute_ensemble_gradcam(model, tensor, target_class=1):
    """
    Hitung Grad-CAM dengan ensemble multi-layer + multi-method.

    Menggunakan:
    - Beberapa layer (blocks[3], blocks[5], blocks[6]) untuk cakupan
      fitur dari low-level hingga high-level
    - Beberapa metode CAM (GradCAM, GradCAM++, LayerCAM) untuk
      robustness yang lebih baik
    - Bobot disesuaikan per-layer dan per-method

    Returns:
        np.ndarray: Grayscale CAM result, normalized [0, 1]
    """
    # Layer yang akan digunakan beserta bobotnya
    # blocks[3] = mid-level features (paling banyak coverage)
    # blocks[5] = mid-high features
    # blocks[6] = high-level features (paling spesifik)
    layer_configs = [
        (model.blocks[3], 0.25),   # Mid-level: banyak coverage spatial
        (model.blocks[5], 0.35),   # Mid-high: keseimbangan coverage+spesifik
        (model.blocks[6], 0.40),   # High-level: paling spesifik
    ]

    # Metode CAM yang akan digunakan beserta bobotnya
    cam_methods = [
        (GradCAM, 0.4),
        (GradCAMPlusPlus, 0.35),
        (LayerCAM, 0.25),
    ]

    targets = [ClassifierOutputTarget(target_class)]
    accumulated = None
    total_weight = 0

    for cam_cls, method_weight in cam_methods:
        for layer, layer_weight in layer_configs:
            try:
                cam = cam_cls(model=model, target_layers=[layer])
                grayscale = cam(input_tensor=tensor, targets=targets)[0]

                weight = method_weight * layer_weight
                if accumulated is None:
                    accumulated = grayscale * weight
                else:
                    accumulated += grayscale * weight
                total_weight += weight
            except Exception as e:
                print(f"   ⚠️ {cam_cls.__name__} on layer failed: {e}")
                continue

    if accumulated is None or total_weight == 0:
        # Fallback ke GradCAM biasa
        cam = GradCAM(model=model, target_layers=[model.blocks[-1]])
        return cam(input_tensor=tensor, targets=targets)[0]

    # Normalize
    result = accumulated / total_weight

    # Enhance contrast menggunakan CLAHE-like approach
    # Ini membantu membuat area lemah lebih terlihat
    p_low, p_high = np.percentile(result, [2, 98])
    if p_high > p_low:
        result = np.clip((result - p_low) / (p_high - p_low), 0, 1)

    return result.astype(np.float32)


# ─── Test-Time Augmentation (TTA) ───────────────────────────────────────────
def predict_with_tta(model, ela_img, original_img):
    """
    Inference dengan Test-Time Augmentation untuk prediksi lebih stabil.

    Menggunakan augmentasi: original, horizontal flip, multi-quality ELA.
    Menggabungkan semua prediksi untuk hasil yang lebih robust.

    Returns:
        prob_tampered: float
        prob_authentic: float
        primary_tensor: tensor untuk Grad-CAM (dari ELA quality 90)
    """
    all_probs = []

    # 1. Prediksi utama (ELA quality 90)
    tensor_main = val_tfm(image=np.array(ela_img))["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor_main)
        probs = torch.softmax(out, 1)[0]
        all_probs.append(probs.cpu().numpy())

    # 2. Horizontal flip
    ela_np = np.array(ela_img)
    ela_flipped = np.fliplr(ela_np).copy()
    tensor_flip = val_tfm(image=ela_flipped)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor_flip)
        probs = torch.softmax(out, 1)[0]
        all_probs.append(probs.cpu().numpy())

    # 3. Multi-quality ELA (quality 75 dan 60)
    for q in [75, 60]:
        ela_q = convert_to_ela_from_pil(original_img, quality=q)
        tensor_q = val_tfm(image=np.array(ela_q))["image"].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(tensor_q)
            probs = torch.softmax(out, 1)[0]
            all_probs.append(probs.cpu().numpy())

    # Rata-rata probabilitas
    avg_probs = np.mean(all_probs, axis=0)
    prob_authentic = float(avg_probs[0])
    prob_tampered = float(avg_probs[1])

    return prob_tampered, prob_authentic, tensor_main


# ─── Deteksi Region Manipulasi ──────────────────────────────────────────────
def detect_manipulation_regions(grayscale_cam, original_img, min_area_ratio=0.003):
    """
    Deteksi region manipulasi dari Grad-CAM heatmap menggunakan
    adaptive thresholding untuk menangkap baik editan halus maupun kasar.

    Args:
        grayscale_cam: Grad-CAM heatmap [H, W] normalized 0-1
        original_img: PIL Image original
        min_area_ratio: Minimum area ratio untuk valid contour

    Returns:
        annotated_img: PIL Image dengan bounding box dan label
        mask_overlay: PIL Image dengan overlay mask transparan
        regions: list of dict dengan info setiap region
    """
    h, w = grayscale_cam.shape

    # ─── Adaptive Thresholding ────────────────────────────────────────
    # Gunakan Otsu's method untuk threshold yang adaptif ke gambar
    cam_uint8 = (grayscale_cam * 255).astype(np.uint8)

    # Otsu threshold
    otsu_thresh, _ = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = otsu_thresh / 255.0

    # Gunakan threshold yang lebih rendah antara Otsu dan fixed
    # Ini memastikan kita tidak kehilangan area yang penting
    adaptive_thresh = min(otsu_val * 0.85, 0.35)
    adaptive_thresh = max(adaptive_thresh, 0.15)  # Minimum threshold

    # Buat binary mask
    mask = (grayscale_cam >= adaptive_thresh).astype(np.uint8) * 255

    # Morphological operations untuk membersihkan noise dan menghubungkan region dekat
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Close dulu (tutup gap) dengan kernel besar
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    # Open (bersihkan noise) dengan kernel kecil
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    # Dilate sedikit untuk memperluas area deteksi
    mask = cv2.dilate(mask, kernel_small, iterations=1)

    # Gaussian blur untuk smoothing
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    # Temukan contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours berdasarkan area minimum
    min_area = int(h * w * min_area_ratio)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Sort by area (terbesar dulu)
    valid_contours.sort(key=cv2.contourArea, reverse=True)

    # Resize original image ke ukuran CAM
    orig_resized = original_img.resize((w, h), Image.LANCZOS)

    # ─── 1. Annotated Image (Bounding boxes + labels) ────────────────
    annotated = orig_resized.copy()
    draw = ImageDraw.Draw(annotated)

    # Coba load font, fallback ke default
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except (IOError, OSError):
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except (IOError, OSError):
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

    regions = []
    colors = [
        (255, 60, 60),    # Merah
        (255, 165, 0),    # Oranye
        (255, 220, 50),   # Kuning
        (220, 80, 220),   # Magenta
        (100, 200, 255),  # Biru muda
    ]

    for i, contour in enumerate(valid_contours[:8]):  # Maks 8 region
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        area_pct = (area / (h * w)) * 100

        # Hitung intensitas rata-rata di region ini
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        mean_intensity = np.mean(grayscale_cam[region_mask > 0])
        severity = "Tinggi" if mean_intensity > 0.6 else "Sedang" if mean_intensity > 0.35 else "Rendah"

        color = colors[i % len(colors)]
        region_id = i + 1

        # Gambar bounding box
        draw.rectangle([x, y, x + bw, y + bh], outline=color, width=2)

        # Gambar corner accents (sudut tebal)
        corner_len = min(12, bw // 4, bh // 4)
        for cx, cy, dx, dy in [
            (x, y, 1, 1), (x + bw, y, -1, 1),
            (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1)
        ]:
            draw.line([(cx, cy), (cx + corner_len * dx, cy)], fill=color, width=3)
            draw.line([(cx, cy), (cx, cy + corner_len * dy)], fill=color, width=3)

        # Label background
        label = f"Area {region_id}"
        sublabel = f"{severity} ({area_pct:.1f}%)"
        bbox_label = draw.textbbox((0, 0), label, font=font_large)
        label_w = bbox_label[2] - bbox_label[0] + 12
        label_h = bbox_label[3] - bbox_label[1] + 8

        label_x = x
        label_y = max(0, y - label_h - 4)

        # Background semi-transparan
        overlay_box = Image.new('RGBA', annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay_box)
        overlay_draw.rounded_rectangle(
            [label_x, label_y, label_x + label_w, label_y + label_h],
            radius=4, fill=(*color, 200)
        )
        annotated = Image.alpha_composite(annotated.convert('RGBA'), overlay_box).convert('RGB')
        draw = ImageDraw.Draw(annotated)

        # Label text
        draw.text((label_x + 6, label_y + 3), label, fill=(255, 255, 255), font=font_large)

        # Sublabel di bawah box
        draw.text((x + 4, y + bh + 4), sublabel, fill=color, font=font_small)

        regions.append({
            "id": region_id,
            "bbox": {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)},
            "area_percent": round(float(area_pct), 2),
            "intensity": round(float(mean_intensity) * 100, 1),
            "severity": severity,
            "color": f"rgb({color[0]},{color[1]},{color[2]})",
        })

    # ─── 2. Mask Overlay (heatmap + contour di atas gambar) ───────────
    orig_np = np.array(orig_resized).astype(np.float32)

    # Buat colored heatmap
    heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Blend dengan graduated alpha berdasarkan intensitas CAM
    alpha_mask = grayscale_cam.copy()
    alpha_mask = cv2.GaussianBlur(alpha_mask, (11, 11), 0)
    # Non-linear alpha: lebih kuat di area tinggi
    alpha_mask = np.power(alpha_mask, 0.7)
    alpha_3ch = np.stack([alpha_mask] * 3, axis=-1)

    blend_strength = 0.65
    blended = orig_np * (1 - alpha_3ch * blend_strength) + heatmap_colored * (alpha_3ch * blend_strength)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Gambar contour outlines
    for i, contour in enumerate(valid_contours[:8]):
        cv2.drawContours(blended, [contour], -1, colors[i % len(colors)], 2)

    mask_overlay = Image.fromarray(blended)

    return annotated, mask_overlay, regions


# ─── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Custom JSON provider untuk handle numpy types
class NumpyJSONProvider(app.json_provider_class):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
app.json = NumpyJSONProvider(app)

@app.route("/")
def index():
    """Serve halaman utama."""
    return send_from_directory("static", "index.html")

@app.route("/api/detect", methods=["POST"])
def detect():
    """
    Endpoint deteksi manipulasi gambar.

    Menerima: multipart/form-data dengan field 'image'
    Mengembalikan JSON berisi hasil prediksi, ELA, dan Grad-CAM.

    Pipeline:
    1. Multi-quality ELA + Test-Time Augmentation untuk prediksi stabil
    2. Multi-layer + Multi-method Grad-CAM ensemble untuk lokalisasi
    3. Adaptive thresholding untuk deteksi region
    """
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang dikirim"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    # Baca dan validasi gambar
    try:
        img_bytes = file.read()
        original = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "File bukan gambar valid"}), 400

    # Simpan sementara untuk ELA
    tmp_name = f"/tmp/detect_{uuid.uuid4().hex}.jpg"
    original.save(tmp_name, "JPEG", quality=95)

    try:
        # ─── ELA ─────────────────────────────────────────────────────────
        ela_img = convert_to_ela(tmp_name, quality=90)

        # ─── Inference dengan TTA ────────────────────────────────────────
        prob_tampered, prob_authentic, tensor = predict_with_tta(
            model, ela_img, original
        )

        threshold = float(request.form.get("threshold", 0.5))
        is_tampered = prob_tampered >= threshold

        # ─── Multi-Layer Ensemble Grad-CAM ───────────────────────────────
        cam_b64 = None
        annotated_b64 = None
        mask_overlay_b64 = None
        regions = []

        try:
            # Ensemble Grad-CAM dari beberapa layer dan metode
            grayscale = compute_ensemble_gradcam(
                model, tensor, target_class=1
            )

            orig_resized = original.resize((IMG_SIZE, IMG_SIZE))
            orig_np = np.array(orig_resized).astype(np.float32) / 255.0
            cam_img = show_cam_on_image(orig_np, grayscale, use_rgb=True)
            cam_pil = Image.fromarray(cam_img)
            cam_b64 = pil_to_base64(cam_pil)

            # ─── Deteksi Region Manipulasi ────────────────────────────
            annotated_img, mask_overlay, regions = detect_manipulation_regions(
                grayscale, original
            )
            annotated_b64 = pil_to_base64(annotated_img)
            mask_overlay_b64 = pil_to_base64(mask_overlay)

        except Exception as e:
            print(f"⚠️ Grad-CAM / Region detection error: {e}")
            import traceback
            traceback.print_exc()

        # ─── ELA multi-quality untuk display ─────────────────────────────
        ela_multi = []
        for q in ELA_QUALITIES:
            ela_q = convert_to_ela(tmp_name, quality=q)
            ela_multi.append(pil_to_base64(ela_q))

        # ─── Response ────────────────────────────────────────────────────
        result = {
            "filename": file.filename,
            "result": "TAMPERED" if is_tampered else "AUTHENTIC",
            "is_tampered": is_tampered,
            "confidence": {
                "tampered": round(prob_tampered * 100, 2),
                "authentic": round(prob_authentic * 100, 2),
            },
            "threshold": threshold * 100,
            "regions": regions,
            "images": {
                "original": pil_to_base64(original),
                "ela": ela_multi[0],       # ELA quality 90 (primary)
                "ela_75": ela_multi[1],     # ELA quality 75
                "ela_60": ela_multi[2],     # ELA quality 60
                "gradcam": cam_b64,
                "annotated": annotated_b64,
                "mask_overlay": mask_overlay_b64,
            }
        }
        return jsonify(result)

    finally:
        # Bersihkan file sementara
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model": "EfficientNet-B4",
        "device": str(DEVICE),
        "model_loaded": model is not None,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
