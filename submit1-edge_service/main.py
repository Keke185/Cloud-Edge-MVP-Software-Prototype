"""
Edge Node Simulation - FastAPI service.

Loads the cloud-generated semantic artifact (distribution.pkl) once at startup,
keeps the capability vectors, decision policy, and the SBERT model resident in
memory, and serves local mismatch-interception requests over HTTP.
"""

import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH", "/data/distribution.pkl")
MODEL_PATH = os.environ.get("MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
NODE_ROLE = os.environ.get("NODE_ROLE", "edge")  # "edge" or "cloud" -- label only, used in responses/logs

#Save status
STATE = {}


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def classify_score(score: float, policy: dict) -> str:
    if score >= policy["aligned_threshold"]:
        return "Aligned"
    elif score >= policy["weakly_aligned_threshold"]:
        return "Weakly Aligned"
    else:
        return policy["mismatched_label"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load artifact + model
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

    print(f"[{NODE_ROLE}] Loading model from {MODEL_PATH}", flush=True)
    STATE["model"] = SentenceTransformer(MODEL_PATH)
    t_model = time.perf_counter()
    model_load_ms = (t_model - t_artifact) * 1000
    total_cold_start_ms = (t_model - t_start) * 1000
    STATE["cold_start_breakdown_ms"] = {
        "artifact_load_ms": round(artifact_load_ms, 1),
        "model_load_ms": round(model_load_ms, 1),
        "total_cold_start_ms": round(total_cold_start_ms, 1),
    }

    print(f"[{NODE_ROLE}] Model loaded in {model_load_ms:.1f}ms. "
          f"COLD-START complete: total={total_cold_start_ms:.1f}ms "
          f"(artifact={artifact_load_ms:.1f}ms + model={model_load_ms:.1f}ms). "
          f"Service ready.", flush=True)

    yield
    #close
    STATE.clear()


app = FastAPI(title="Semantic Matching Edge Service", lifespan=lifespan)


class MatchRequest(BaseModel):
    item_text: str = Field(..., min_length=1, description="Recruitment evaluation item text (kept local; never persisted)")
    item_id: Optional[str] = Field(default=None, description="Optional caller-supplied identifier for correlation/logging only")


class MatchResponse(BaseModel):
    item_id: Optional[str]
    role_id: str
    predicted_label: str
    similarity_score: float
    best_capability_block: int
    decision_policy: dict
    artifact_version: str
    node_role: str
    embedding_ms: float
    similarity_ms: float
    total_processing_ms: float


@app.get("/health")
def health():
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Service not ready: artifact/model not loaded yet")
    return {
        "status": "ok",
        "node_role": NODE_ROLE,
        "role_id": STATE["role_id"],
        "capability_blocks": int(STATE["capability_vectors"].shape[0]),
        "artifact_version": STATE["artifact_metadata"].get("artifact_version"),
    }


@app.get("/startup_timing")
def startup_timing():
    # Query cold start logs
    if "cold_start_breakdown_ms" not in STATE:
        raise HTTPException(status_code=503, detail="Startup not complete yet")
    return {"node_role": NODE_ROLE, **STATE["cold_start_breakdown_ms"]}


@app.post("/edge/init")
def edge_init():
    # Initialize artifact
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail= "Artifact/model not loaded yet")

    return {
        "status": "initialised",
        "node_role": NODE_ROLE,
        "role_id": STATE["role_id"],
        "capability_blocks": int(STATE["capability_vectors"].shape[0]),
    }


@app.post("/match/check", response_model=MatchResponse)
def match_check(req: MatchRequest):

    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Service not ready: artifact/model not loaded yet")

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

    predicted_label = classify_score(score, STATE["decision_policy"])
    total_ms = (time.perf_counter() - t_start) * 1000


    return MatchResponse(
        item_id=req.item_id,
        role_id=STATE["role_id"],
        predicted_label=predicted_label,
        similarity_score=score,
        best_capability_block=best_block,
        decision_policy=STATE["decision_policy"],
        artifact_version=STATE["artifact_metadata"].get("artifact_version", "unknown"),
        node_role=NODE_ROLE,
        embedding_ms=embedding_ms,
        similarity_ms=similarity_ms,
        total_processing_ms=total_ms,
    )