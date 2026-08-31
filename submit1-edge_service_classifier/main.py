import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

#Load file
ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "/data/distribution.pkl")
MODEL_PATH = os.environ.get("MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
NODE_ROLE = os.environ.get("NODE_ROLE", "edge")
NEW_LOGIC_PATH = os.environ.get("NEW_LOGIC_PATH", "/data/new_logic_pipeline.pkl")

#Save Status
STATE = {}

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def build_features(sim_vec):
    mean_sim = sim_vec.mean()
    std_sim = sim_vec.std()
    sorted_sim = np.sort(sim_vec)[::-1]
    margin = sorted_sim[0] - sorted_sim[1]
    return np.concatenate([sim_vec, [mean_sim, std_sim, margin]])


def new_logic(sim_vec, scaler, clf, mismatched_label):
    feat = build_features(sim_vec).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred = int(clf.predict(feat_scaled)[0])
    return {0: "Aligned", 1: "Weakly Aligned", 2: mismatched_label}.get(pred, mismatched_label)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t_start = time.perf_counter()

    print(f"[{NODE_ROLE}] COLD-START begin. Loading artifact from {ARTIFACT_PATH}", flush=True)

    with open(ARTIFACT_PATH, "rb") as f:
        artifact = pickle.load(f)

    STATE["role_id"] = artifact["role_id"]
    STATE["capability_vectors"] = np.asarray(artifact["capability_vectors"])
    STATE["capability_chunks"] = artifact["capability_chunks"]
    STATE["decision_policy"] = artifact["decision_policy"]
    STATE["artifact_metadata"] = artifact["metadata"]

    t_artifact = time.perf_counter()
    artifact_load_ms = (t_artifact - t_start) * 1000

    print(f"[{NODE_ROLE}] Artifact loaded in {artifact_load_ms:.1f}ms. role_id={STATE['role_id']}, "
          f"capability_blocks={STATE['capability_vectors'].shape[0]}", flush=True)

    print(f"[{NODE_ROLE}] Loading new-logic pipeline from {NEW_LOGIC_PATH}", flush=True)
    with open(NEW_LOGIC_PATH, "rb") as f:

        pipeline = pickle.load(f)
    STATE["new_logic_scaler"] = pipeline["scaler"]
    STATE["new_logic_clf"] = pipeline["clf"]
    t_pipeline = time.perf_counter()
    pipeline_load_ms = (t_pipeline - t_artifact) * 1000

    print(f"[{NODE_ROLE}] Classifier pipeline loaded in {pipeline_load_ms:.1f}ms", flush=True)

    print(f"[{NODE_ROLE}] Loading semantic model from {MODEL_PATH}", flush=True)

    STATE["model"] = SentenceTransformer(MODEL_PATH)

    t_model = time.perf_counter()
    model_load_ms = (t_model - t_pipeline) * 1000
    total_cold_start_ms = (t_model - t_start) * 1000

    STATE["cold_start_breakdown_ms"] = {
        "artifact_load_ms": round(artifact_load_ms, 1),
        "pipeline_load_ms": round(pipeline_load_ms, 1),
        "model_load_ms": round(model_load_ms, 1),
        "total_cold_start_ms": round(total_cold_start_ms, 1),
    }

    print(f"[{NODE_ROLE}] Model loaded in {model_load_ms:.1f}ms "
          f"COLD-START complete: total={total_cold_start_ms:.1f}ms. Service ready (SIMILARITY-PROFILE CLASSIFIER)",
          flush=True)

    yield
    STATE.clear()


app = FastAPI(title="Semantic Matching Edge Service (Similarity-Profile Classifier)", lifespan=lifespan)


class MatchRequest(BaseModel):
    item_text: str = Field(..., min_length=1)
    item_id: Optional[str] = Field(default=None)


class MatchResponse(BaseModel):
    item_id: Optional[str]
    role_id: str
    predicted_label: str
    similarity_score: float
    best_capability_block: int
    decision_logic: str
    artifact_version: str
    node_role: str
    embedding_ms: float
    similarity_ms: float
    total_processing_ms: float


@app.get("/health")
def health():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Service not ready: artifact/model/pipeline not loaded yet")
    return {
        "status": "ok",
        "node_role": NODE_ROLE,
        "role_id": STATE["role_id"],
        "capability_blocks": int(STATE["capability_vectors"].shape[0]),
        "artifact_version": STATE["artifact_metadata"].get("artifact_version"),
        "decision_logic": "similarity_profile_classifier",
    }

@app.get("/startup_timing")
def startup_timing():

    if "cold_start_breakdown_ms" not in STATE:

        raise HTTPException(status_code=503, detail="Startup not complete yet")

    return {"node_role": NODE_ROLE, **STATE["cold_start_breakdown_ms"]}


@app.post("/edge/init")
def edge_init():

    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Artifact/model/pipeline not loaded yet")

    return {
        "status": "initialised",
        "node_role": NODE_ROLE,
        "role_id": STATE["role_id"],
        "capability_blocks": int(STATE["capability_vectors"].shape[0]),
    }


@app.post("/match/check", response_model=MatchResponse)
def match_check(req: MatchRequest):

    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Service not ready: artifact/model/pipeline not loaded yet")

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    item_vector = STATE["model"].encode([req.item_text])
    t1 = time.perf_counter()

    embedding_ms = (t1 - t0) * 1000

    t0 = time.perf_counter()
    sim = cosine_similarity_matrix(np.asarray(item_vector), STATE["capability_vectors"])[0]
    score = float(sim.max())
    best_block = int(sim.argmax())
    t1 = time.perf_counter()

    similarity_ms = (t1 - t0) * 1000

    predicted_label = new_logic(sim, STATE["new_logic_scaler"], STATE["new_logic_clf"],
                                 STATE["decision_policy"]["mismatched_label"])
    total_ms = (time.perf_counter() - t_start) * 1000

    return MatchResponse(
        item_id=req.item_id,
        role_id=STATE["role_id"],
        predicted_label=predicted_label,
        similarity_score=score,
        best_capability_block=best_block,
        decision_logic="similarity_profile_classifier",
        artifact_version=STATE["artifact_metadata"].get("artifact_version", "unknown"),
        node_role=NODE_ROLE,
        embedding_ms=embedding_ms,
        similarity_ms=similarity_ms,
        total_processing_ms=total_ms,
    )
