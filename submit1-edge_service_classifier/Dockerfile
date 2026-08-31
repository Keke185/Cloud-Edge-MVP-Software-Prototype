# Edge Node - Similarity Profile Classifier Configuration.

# submit1-edge_service, maximum similarity double threshold rule).

# # Load two artifacts from the shared "artifact-store" volume at startup:

# - distribution.pkl (from submit1-cloud_service)

# - new_logic_pipeline.pkl (from submit1-cloud_service-classifier)

# Neither of these artifacts is integrated into the image, so this Dockerfile only needs to install

# dependencies (including scikit-learn for runtime deserialization of StandardScaler/LogisticRegression objects) and copy main.py.

# Build (run from the folder containing this Dockerfile, main.py, and requirements.txt):

# docker build -t vcse-edge-classifier .

# Run: docker run --rm -p 8002:8000 -v artifact-store:/data \ -e NODE_ROLE=edge-classifier vcse-edge-classifier

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV ARTIFACT_PATH=/data/distribution.pkl
ENV NEW_LOGIC_PATH=/data/new_logic_pipeline.pkl
ENV MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
ENV NODE_ROLE=edge-classifier

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
