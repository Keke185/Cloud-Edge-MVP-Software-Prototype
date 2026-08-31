# docker build -t vcse-cloud .
# Run (writes distribution.pkl into a named volume "artifact-store"):
# docker run --rm -v artifact-store:/data vcse-cloud

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_artifact.py .

ENV OUTPUT_DIR=/data
VOLUME ["/data"]

CMD ["python", "generate_artifact.py"]
