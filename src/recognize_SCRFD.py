import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ======================================================
# PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_DIR   = os.path.join(BASE_DIR, "embeddings")
TEST_DIR = os.path.join(BASE_DIR, "data", "test_inputs")
OUT_DIR  = os.path.join(BASE_DIR, "outputs_yolo")

os.makedirs(OUT_DIR, exist_ok=True)

EMB_NPY = os.path.join(DB_DIR, "embeddings.npy")
LBL_NPY = os.path.join(DB_DIR, "labels.npy")

# ======================================================
# PARAMETERS (STABLE VALUES)
# ======================================================
SIM_THRESHOLD     = 0.15   # final accept
RECHECK_THRESHOLD = 0.05  # weak face recheck
IOU_THRESHOLD     = 0.4

GAMMA_VALUES      = [0.7, 0.9, 1.0, 1.2, 1.4]
SCALE_FACTORS     = [0.75, 1.0, 1.25]

TEMPORAL_ALPHA    = 0.6      # smoothing memory

# ======================================================
# LOAD DATABASE
# ======================================================
print("[INFO] Loading face database...")
db_embeddings = np.load(EMB_NPY)
db_labels     = np.load(LBL_NPY)

db_embeddings = db_embeddings / np.linalg.norm(
    db_embeddings, axis=1, keepdims=True
)

print(f"[INFO] Loaded {len(db_labels)} identities")

# ======================================================
# INIT INSIGHTFACE (SCRFD + ArcFace)
# ======================================================
print("[INFO] Initializing SCRFD + ArcFace...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(1024, 1024))

# ======================================================
# IMAGE ENHANCEMENT
# ======================================================
def gamma_correct(img, gamma):
    inv = 1.0 / gamma
    table = np.array(
        [(i / 255.0) ** inv * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(img, table)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# ======================================================
# IOU + MERGE
# ======================================================
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)

def merge_faces(faces):
    merged = []
    for f in faces:
        keep = True
        for m in merged:
            if iou(f.bbox, m.bbox) > IOU_THRESHOLD:
                keep = False
                break
        if keep:
            merged.append(f)
    return merged

# ======================================================
# TEMPORAL MEMORY
# ======================================================
identity_scores = defaultdict(float)
identity_ids = {}
next_id = 1

# ======================================================
# PROCESS ALL TEST IMAGES
# ======================================================
for img_name in sorted(os.listdir(TEST_DIR)):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    print(f"\n[INFO] Processing: {img_name}")

    img_path = os.path.join(TEST_DIR, img_name)
    base_img = cv2.imread(img_path)
    if base_img is None:
        continue

    detected_faces = []

    # ----------------------------------------------
    # MULTI-SCALE + GAMMA + CLAHE SWEEP
    # ----------------------------------------------
    for scale in SCALE_FACTORS:
        resized = cv2.resize(
            base_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )

        for gamma in GAMMA_VALUES:
            enhanced = gamma_correct(resized, gamma)
            enhanced = apply_clahe(enhanced)

            faces = app.get(enhanced)

            # scale boxes back
            for f in faces:
                f.bbox /= scale
                detected_faces.append(f)

    detected_faces = merge_faces(detected_faces)
    print(f"[INFO] Faces detected: {len(detected_faces)}")

    # ----------------------------------------------
    # RECOGNITION
    # ----------------------------------------------
    for face in detected_faces:

        x1, y1, x2, y2 = face.bbox.astype(int)
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        sims = cosine_similarity(
            emb.reshape(1, -1),
            db_embeddings
        )[0]

        idx = np.argmax(sims)
        score = sims[idx]
        label = db_labels[idx]

        # Temporal smoothing
        prev = identity_scores[label]
        smooth = TEMPORAL_ALPHA * prev + (1 - TEMPORAL_ALPHA) * score
        identity_scores[label] = max(identity_scores[label], smooth)

        if label not in identity_ids:
            identity_ids[label] = next_id
            next_id += 1

        pid = identity_ids[label]

        # Decision
        if smooth >= SIM_THRESHOLD:
            color = (0, 255, 0)
            text = f"ID:{pid} {label} ({smooth:.2f})"
        elif score >= RECHECK_THRESHOLD:
            color = (0, 165, 255)
            text = f"ID:{pid}? {label} ({score:.2f})"
        else:
            color = (0, 0, 255)
            text = "UNKNOWN"

        cv2.rectangle(base_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            base_img,
            text,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, base_img)
    print(f"✅ Saved: {out_path}")

print("\n🎯 ALL IMAGES PROCESSED SUCCESSFULLY")
