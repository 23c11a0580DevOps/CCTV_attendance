import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from insightface.app import FaceAnalysis

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ALIGNED_DIR = os.path.join(BASE_DIR, "processed", "aligned_faces")
EMB_DIR     = os.path.join(BASE_DIR, "embeddings")

os.makedirs(ALIGNED_DIR, exist_ok = True)
os.makedirs(EMB_DIR, exist_ok=True)

EMB_NPY = os.path.join(EMB_DIR, "embeddings.npy")
LBL_NPY = os.path.join(EMB_DIR, "labels.npy")
PKL_DB  = os.path.join(EMB_DIR, "face_db.pkl")

# ======================================================
# INIT ARCFACE (InsightFace)
# ======================================================
print("[INFO] Initializing ArcFace model...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1)

# ======================================================
# GENERATE EMBEDDINGS
# ======================================================
embeddings = []
labels = []

print("[INFO] Generating embeddings from aligned faces...")

for img_name in tqdm(os.listdir(ALIGNED_DIR)):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(ALIGNED_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # ArcFace expects BGR (OpenCV default)
    faces = app.get(img)

    if len(faces) == 0:
        print(f"[WARN] No face detected in {img_name}")
        continue

    # Exactly ONE face per aligned reference image
    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)  # normalize

    embeddings.append(emb)
    labels.append(os.path.splitext(img_name)[0])

# ======================================================
# SAVE DATABASE
# ======================================================
if len(embeddings) == 0:
    raise RuntimeError("[x] No embeddings generated. Check aligned faces.")

embeddings = np.array(embeddings)
labels = np.array(labels)

np.save(EMB_NPY, embeddings)
np.save(LBL_NPY, labels)

with open(PKL_DB, "wb") as f:
    pickle.dump({
        "embeddings": embeddings,
        "labels": labels
    }, f)

print("[OK] ArcFace embedding generation completed")
print("[DIR] Embeddings saved to:", EMB_DIR)
print("[NUM] Total identities enrolled:", len(labels))
