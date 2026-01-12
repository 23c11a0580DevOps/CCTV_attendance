import cv2
import os
import json
import numpy as np
from tqdm import tqdm

# ======================================================
# BASE DIRECTORY
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(
    BASE_DIR, "data", "pics"
)

OUT_FACE_DIR = os.path.join(BASE_DIR, "processed", "aligned_faces")
OUT_NPY_DIR  = os.path.join(BASE_DIR, "processed", "faces_npy")
OUT_JSON_DIR = os.path.join(BASE_DIR, "processed", "faces_json")

os.makedirs(OUT_FACE_DIR, exist_ok=True)
os.makedirs(OUT_NPY_DIR, exist_ok=True)
os.makedirs(OUT_JSON_DIR, exist_ok=True)

print("[INFO] INPUT_DIR =", INPUT_DIR)
print("[INFO] Images found =", len(os.listdir(INPUT_DIR)))

# ======================================================
# PROCESS REFERENCE IMAGES (HTNO PHOTOS)
# ======================================================
for img_name in tqdm(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # HTNO images → face occupies entire image
    face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))

    # ArcFace normalization
    face_tensor = face_resized.astype(np.float32)
    face_tensor = (face_tensor - 127.5) / 127.5
    face_tensor = np.transpose(face_tensor, (2, 0, 1))  # CHW

    base = os.path.splitext(img_name)[0]

    # Save files
    np.save(os.path.join(OUT_NPY_DIR, base + ".npy"), face_tensor)

    cv2.imwrite(
        os.path.join(OUT_FACE_DIR, base + ".jpg"),
        cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)
    )

    meta = {
        "filename": img_name,
        "bbox": [0, 0, w, h],
        "landmarks_5": None,
        "identity": base,
        "note": "HTNO reference image, direct ArcFace preprocessing"
    }

    with open(os.path.join(OUT_JSON_DIR, base + ".json"), "w") as f:
        json.dump(meta, f, indent=4)

print("[OK] Reference face preprocessing completed successfully")
