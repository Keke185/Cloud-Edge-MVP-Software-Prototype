"""
Cloud Layer - Similarity-Profile Classifier Pipeline Generator.

decision-overhead microbenchmark.
which fitted the scaler/classifier on random synthetic data purely to measure
inference latency, not to produce a meaningful classifier. That placeholder
is fine for a timing test, but is NOT appropriate to deploy as part of the MVP demo, because it would return effectively random predictions.
"""

import csv
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_items_final.csv")


CAPABILITY_CHUNKS = [
    "Enterprise Video Conferencing & Signaling Architecture: Ability to design and deploy enterprise-grade video conferencing architectures, with strong expertise in large-scale, high-concurrency audio/video system topologies and network architecture design. Proficient in media streaming transport and signaling control protocols, with in-depth knowledge of classical multimedia framework protocols such as ITU-T H.323, including H.225, H.245, Q.931, and H.235 security encryption, and IETF SIP, including SDP-based media negotiation and call routing orchestration based on standard SIP Trunking. Deep understanding of the real-time communication (RTC) technology stack and multiple application scenarios, including real-time conversational platforms and interactive live streaming. Hands-on command of WebRTC, RTP/RTCP, SRTP encrypted transport, transport-layer TCP/UDP, media stream encapsulation and forwarding, including RTSP, RTMP, and standard Socket-based communication, as well as integration with traditional telecommunications networks such as SIP/PSTN. Able to distinguish the architectural trade-offs between low-latency live streaming and real-time conferencing.",
    "Conference Management Systems & Business Control: Familiar with the underlying control logic of conference management systems, with the ability to efficiently allocate maximum concurrent media ports, large-scale user resources, virtual meeting room provisioning, and conference control policies. Proficient in enterprise audio/video call control and control-flow management, including URI-based calling, direct IP dialing, interactive voice response (IVR), and diversified meeting access modes such as virtual meeting rooms (VMRs).",
    "MCU & Media Engine Core Concepts: Advanced knowledge of Multipoint Control Units (MCUs) and media engine processing architectures. Deep understanding of the core logic and server-resource trade-offs between full encoding/decoding-based audio mixing and video compositing, namely AVC with centralised transcoding and composition, and Scalable Video Coding (SVC) based multi-stream distribution with selective forwarding. Familiar with multi-level cascading across media servers and dynamic channel multiplexing. Capable of designing and optimising media resource pooling, distributed clustering, disaster recovery, high availability, and load balancing.",
    "Video Endpoints & Room-Based Systems: Familiar with the engineering principles and signal transmission mechanisms of various hardware video endpoints and room-based collaboration devices. Expertise in room-based conferencing systems, with solid experience in signal-chain design for multi-channel video input/output interfaces, including HDMI, composite video, XLR, RCA, optical fiber, and optical-electrical transmission interfaces, as well as integration of peripherals such as PoE touch panels, serial control interfaces, and wireless screen sharing systems. Proficient in meeting-room automation control, including voice-activated dynamic switching, multi-view presentation, auto layout, and common multi-window layout combinations.",
    "4K UHD Video, HD Codecs & Multi-Party Media Processing: Capable of delivering 4K UHD video conferencing solutions, with expertise in multi-party media processing and parameter tuning under high-definition and low-bandwidth constraints. Skilled in weak-network QoS assurance mechanisms such as audio/video synchronization (A/V sync) and jitter buffering. Proficient in mainstream high-definition encoding and decoding protocols, including video codecs (H.265/HEVC for 4K, H.264 High/Base Profile, simultaneous encoding and decoding for live video streams and presentation content with dual-stream transmission capability supported by H.239 and BFCP dual-stream protocols) and audio codecs (communication and broadcast-grade encoding & decoding standards such as Opus, AAC, AAC-LD, G.711a/u, G.722, G.729). Proficient in advanced audio processing and 3A algorithms, including acoustic echo cancellation (AEC), automatic noise suppression (ANS), and automatic gain control (AGC). Strong expertise in multi-channel stereo, spatial audio, multi-channel wideband voice mixing, and dynamic subtitle overlay based on real-time automatic speech recognition (ASR).",
    "Cloud Communications (CPaaS) & Conversational AI Pipelines: Proficient in the Cloud Communications Platform as a Service (CPaaS) delivery model, with the ability to use standardized SDKs and REST APIs to rapidly embed real-time voice, video, interactive live streaming, and instant messaging/chat capabilities into Web applications, mobile platforms including iOS, Android, and cross-platform frameworks, and IoT devices. Keeps pace with generative AI trends, with knowledge of cutting-edge Conversational AI, Voice AI, and agentic architectures. Familiar with large language model (LLM) orchestration, bidirectional ASR/TTS speech conversion, the open Model Context Protocol (MCP, an emerging industry de facto standard), and function-calling frameworks. Capable of designing low-latency, intelligent, human-machine real-time audio/video interaction applications.",
]

LABEL_TO_CODE = {"Aligned": 0, "Weakly Aligned": 1, "Mismatched": 2}


def build_features(sim_vec: np.ndarray) -> np.ndarray:
    mean_sim = sim_vec.mean()
    std_sim = sim_vec.std()
    sorted_sim = np.sort(sim_vec)[::-1]
    margin = sorted_sim[0] - sorted_sim[1]
    return np.concatenate([sim_vec, [mean_sim, std_sim, margin]])


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def main():
    print("*" * 50)
    print("  Similarity-Profile Classifier - Pipeline Generator")
    print("*" * 50)

    output_dir = os.environ.get("OUTPUT_DIR", "/data")
    os.makedirs(output_dir, exist_ok=True)

    model_id = os.environ.get("MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"\n Load Model: {model_id}")
    model = SentenceTransformer(model_id)
    print("Model loaded")

    print(f"\n Reading benchmark dataset: {DATASET_PATH}")
    with open(DATASET_PATH, encoding="cp1252") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} items loaded")

    item_texts = [r["item_text"] for r in rows]
    gold_labels = [r["gold_label"] for r in rows]
    unknown = sorted(set(gold_labels) - set(LABEL_TO_CODE))
    if unknown:
        raise ValueError(f"Unrecognised gold_label value(s) in dataset: {unknown}")
    y = np.array([LABEL_TO_CODE[lbl] for lbl in gold_labels])

    print("\n Encoding capability block")
    capability_vectors = model.encode(CAPABILITY_CHUNKS, show_progress_bar=False)
    item_vectors = model.encode(item_texts, show_progress_bar=True)

    print("\n Calculate block-level similarity profile and construct features")
    sim_matrix = cosine_similarity_matrix(np.asarray(item_vectors), np.asarray(capability_vectors))
    X = np.array([build_features(row) for row in sim_matrix])
    print(f" Feature matrix: {X.shape} (6 similarities + mean + std + margin)")

    print("\n Fitting StandardScaler + multinomial LogisticRegression on the full 200-item benchmark")
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial").fit(scaler.transform(X), y)
    train_acc = clf.score(scaler.transform(X), y)
    print(f" Fit complete. Full-data training accuracy: {train_acc:.4f}")
    print("     (This is a full-data fit for deployment, not a held-out estimate. "
          "The cross-validated performance numbers reported in the dissertation "
          "come from experiment_rigorous_cv_v3_leakfixed.py, not from this script.)")

    output_path = os.path.join(output_dir, "new_logic_pipeline.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf}, f)

    print(f"\n{'*' * 55}")
    print("  Classifier pipeline ready.")
    print(f"     File path : {os.path.abspath(output_path)}")
    print(f"{'*' * 55}")


if __name__ == "__main__":
    main()
