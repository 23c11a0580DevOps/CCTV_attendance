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

EMB_DIR = os.path.join(BASE_DIR, "embeddings")
TEST_DIR = os.path.join(BASE_DIR, "data", "test_inputs")
OUT_DIR = os.path.join(BASE_DIR, "outputs_final")

os.makedirs(OUT_DIR, exist_ok=True)

EMB_NPY = os.path.join(EMB_DIR, "embeddings.npy")
LBL_NPY = os.path.join(EMB_DIR, "labels.npy")

# ======================================================
# PARAMETERS
# ======================================================
SIM_THRESHOLD = 0.20
RECHECK_THRESHOLD = 0.40
IOU_THRESHOLD = 0.4

GAMMA_VALUES = [0.8, 1.0, 1.3]   # gamma sweep
TEMPORAL_ALPHA = 0.6            # smoothing factor

# ======================================================
# LOAD DATABASE
# ======================================================
print("[INFO] Loading face database...")
db_embeddings = np.load(EMB_NPY)
db_labels = np.load(LBL_NPY)

db_embeddings = db_embeddings / np.linalg.norm(
    db_embeddings, axis=1, keepdims=True
)

print(f"[INFO] Loaded {len(db_labels)} enrolled identities")

# ======================================================
# INIT INSIGHTFACE
# ======================================================
print("[INFO] Initializing RetinaFace + ArcFace...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=-1, det_size=(1024, 1024))

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def gamma_correct(image, gamma):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def enhance_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
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
# GLOBAL TRACKERS
# ======================================================
identity_scores = defaultdict(float)   # best confidence per identity
identity_ids = {}                      # identity → ID
next_id = 1

# ======================================================
# PROCESS IMAGES
# ======================================================
for img_name in os.listdir(TEST_DIR):

    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    print(f"\n[INFO] Processing: {img_name}")

    img_path = os.path.join(TEST_DIR, img_name)
    base_img = cv2.imread(img_path)

    if base_img is None:
        continue

    detected_faces = []

    # ==================================================
    # GAMMA + CLAHE SWEEP
    # ==================================================
    for gamma in GAMMA_VALUES:
        img = gamma_correct(base_img, gamma)
        img = enhance_clahe(img)

        faces = app.get(img)
        detected_faces.extend(faces)

    detected_faces = merge_faces(detected_faces)
    print(f"[INFO] Faces detected after sweep: {len(detected_faces)}")

    # ==================================================
    # RECOGNITION + TEMPORAL SMOOTHING
    # ==================================================
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
        smooth_score = TEMPORAL_ALPHA * prev + (1 - TEMPORAL_ALPHA) * score
        identity_scores[label] = max(identity_scores[label], smooth_score)

        # Assign ID
        if label not in identity_ids:
            identity_ids[label] = next_id
            next_id += 1

        person_id = identity_ids[label]

        # False-positive recheck
        if smooth_score < SIM_THRESHOLD:
            if score < RECHECK_THRESHOLD:
                label = "UNKNOWN"
                person_id = "-"
                color = (0, 0, 255)
            else:
                color = (0, 165, 255)  # uncertain
        else:
            color = (0, 255, 0)

        text = f"ID:{person_id} {label} ({smooth_score:.2f})"

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

print("\n🎯 Processing complete.")
