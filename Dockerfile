# Chess vs Hybrid RL Model — deployment image (Hugging Face Spaces / any Docker host)
# HF Spaces convention: the app must listen on port 7860.

FROM python:3.11-slim

WORKDIR /code

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Core engine modules from the repo root
COPY AlphaZeroNetwork.py MCTS.py encoder.py ./
# Web app, piece images, and the HPC-trained model
COPY app ./app
COPY images ./images
COPY weights/HPC_20x256.pt ./weights/HPC_20x256.pt

ENV DEPLOYED=1
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app/server.py"]
