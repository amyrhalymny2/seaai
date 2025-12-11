import io
import re
import os
import tempfile
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from annoy import AnnoyIndex

# -------------------------
# Config / Globals
# -------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL = None  # lazy load
COMMON_WORDS = [
    "شركة", "شركه", "المحدودة", "محدودة", "ذ.م.م", "ذ.م.م.", "limited", "ltd",
    "مساهمة", "ش.م.م", "ltd.", "co", "company", "for", "and", "&"
]

app = FastAPI(title="Company-Dedupe-API", version="1.0")


# -------------------------
# Helpers
# -------------------------
def lazy_load_model():
    global MODEL
    if MODEL is None:
        MODEL = SentenceTransformer(MODEL_NAME)
    return MODEL


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[ًٌٍَُِّٰٰ]", "", s)  # basic diacritics remove
    s = re.sub(r"[^\w\s\u0600-\u06FF]", " ", s)  # keep Arabic letters, numbers, underscores, spaces
    s = re.sub(r"\s+", " ", s).strip()
    words = [w for w in s.split() if w not in COMMON_WORDS]
    return " ".join(words)


def compute_embeddings(names: List[str], batch_size: int = 64, force_recompute: bool = False):
    model = lazy_load_model()
    # SentenceTransformer.encode supports batching
    embeddings = model.encode(names, convert_to_numpy=True, batch_size=batch_size, show_progress_bar=False)
    embeddings = normalize(embeddings)
    return embeddings


def build_annoy_index(embeddings: np.ndarray, n_trees: int = 10, metric: str = "angular"):
    dim = embeddings.shape[1]
    index = AnnoyIndex(dim, metric=metric)
    for i, v in enumerate(embeddings):
        index.add_item(i, v)
    index.build(n_trees)
    return index


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_with_annoy(embeddings: np.ndarray, index: AnnoyIndex, threshold: float = 0.85, top_k: int = 10):
    n = embeddings.shape[0]
    uf = UnionFind(n)
    # Annoy returns distances (depends on metric). For angular: dist in [0..2], cos ~ 1 - dist^2 / 2 (approx)
    for i in range(n):
        ids, dists = index.get_nns_by_item(i, top_k + 1, include_distances=True)
        for nid, dist in zip(ids[1:], dists[1:]):  # skip self
            # convert angular distance -> cosine similarity approx
            try:
                cos_sim = 1 - (dist ** 2) / 2
            except Exception:
                cos_sim = np.dot(embeddings[i], embeddings[nid])
            if cos_sim >= threshold:
                uf.union(i, nid)
    groups = {}
    for i in range(n):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)
    clusters = [members for members in groups.values() if len(members) > 1]
    return clusters


# -------------------------
# Endpoints
# -------------------------

@app.post("/cluster-file", summary="Upload CSV file and receive clustered CSV")
async def cluster_file(
    file: UploadFile = File(...),
    col: str = Form("Name"),
    threshold: float = Form(0.85),
    batch: int = Form(64),
    k: int = Form(10),
    trees: int = Form(10),
):
    # validate file type
    if not file.filename.endswith((".csv", ".CSV")):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    # read CSV into pandas
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if col not in df.columns:
        raise HTTPException(status_code=400, detail=f"CSV has no column '{col}'. Available columns: {list(df.columns)}")

    names_raw = df[col].fillna("").astype(str).tolist()
    names_norm = [normalize_name(x) for x in names_raw]
    indices = [i for i, n in enumerate(names_norm) if n.strip() != ""]
    names_to_process = [names_norm[i] for i in indices]

    if len(names_to_process) == 0:
        # nothing to cluster
        df["cluster_id"] = -1
        out = df.to_csv(index=False).encode("utf-8")
        return StreamingResponse(io.BytesIO(out), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename=clusters_{file.filename}"})

    # compute embeddings
    embeddings = compute_embeddings(names_to_process, batch_size=batch)

    # build annoy
    index = build_annoy_index(embeddings, n_trees=trees)

    # cluster
    clusters = cluster_with_annoy(embeddings, index, threshold=threshold, top_k=k)

    # map back cluster ids
    cluster_id_col = [-1] * len(names_raw)
    for cid, members in enumerate(clusters, start=1):
        for idx in members:
            original_idx = indices[idx]
            cluster_id_col[original_idx] = cid

    df["cluster_id"] = cluster_id_col

    out_bytes = df.to_csv(index=False).encode("utf-8")
    return StreamingResponse(io.BytesIO(out_bytes), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename=clusters_{file.filename}"})


@app.post("/cluster-json", summary="Send JSON with 'names': [..] and get clusters JSON")
async def cluster_json(
    payload: dict
):
    """
    Example payload:
    {
      "names": ["شركة الزاهد ...", "شركه الزاهد ...", "Other Co"],
      "threshold": 0.85,
      "batch": 64,
      "k": 10,
      "trees": 10
    }
    """
    if "names" not in payload or not isinstance(payload["names"], list):
        raise HTTPException(status_code=400, detail="Payload must contain 'names': list")

    names_raw = [str(x) for x in payload["names"]]
    threshold = float(payload.get("threshold", 0.85))
    batch = int(payload.get("batch", 64))
    k = int(payload.get("k", 10))
    trees = int(payload.get("trees", 10))

    names_norm = [normalize_name(x) for x in names_raw]
    indices = [i for i, n in enumerate(names_norm) if n.strip() != ""]
    names_to_process = [names_norm[i] for i in indices]

    if len(names_to_process) == 0:
        return JSONResponse({"clusters": []})

    embeddings = compute_embeddings(names_to_process, batch_size=batch)
    index = build_annoy_index(embeddings, n_trees=trees)
    clusters_idx = cluster_with_annoy(embeddings, index, threshold=threshold, top_k=k)

    # Translate cluster indices back to original names
    clusters = []
    for members in clusters_idx:
        clusters.append([names_raw[indices[m]] for m in members])

    return JSONResponse({"clusters": clusters})


@app.get("/health")
def health():
    return {"status": "ok"}
