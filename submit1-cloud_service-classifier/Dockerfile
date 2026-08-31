# Cloud Layer - Similarity-Profile Classifier Pipeline Generator image.
# Build (run from the folder containing this Dockerfile, fit_new_logic_pipeline.py,
# requirements.txt, and dataset_items_final.csv):
#   docker build -t vcse-classifier-fit .
# Run (writes new_logic_pipeline.pkl into a named volume "artifact-store"):
#   docker run --rm -v artifact-store:/data vcse-classifier-fit

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fit_new_logic_pipeline.py .
COPY dataset_items_final.csv .

ENV OUTPUT_DIR=/data
VOLUME ["/data"]

CMD ["python", "fit_new_logic_pipeline.py"]
