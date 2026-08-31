# Build (run from the folder containing this Dockerfile, main.py, and requirements.txt
#   docker build -t vcse-edge .

# Run - unsandboxed
#   docker run --rm -p 8001:8000 -v artifact-store:/data \     -e NODE_ROLE=edge-unsandboxed vcse-edge
#
# Run - sandboxed
#   docker run --rm -p 8002:8000 -v artifact-store:/data \   -e NODE_ROLE=edge-sandboxed \   --cpus="0.5" --memory="256m" --network=none vcse-edge


FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV ARTIFACT_PATH=/data/distribution.pkl
ENV MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
ENV NODE_ROLE=edge

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
