FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4

COPY models/ models/
COPY src/predict_cli.py src/predict_cli.py
COPY results/ results/

ENTRYPOINT ["python", "src/predict_cli.py"]