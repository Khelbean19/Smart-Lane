FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch (much smaller, no GPU on free tier); versions match uv.lock
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
# ultralytics pulls in the GUI build opencv-python, which shares the cv2 package
# with opencv-python-headless. Remove it, then reinstall headless so cv2 is intact.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps "opencv-python-headless~=5.0.0" \
    && python -c "import cv2; print('OpenCV', cv2.__version__)"

COPY --chown=user . .
RUN chmod +x start.sh
USER user

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    VISION_WORKER_HOST=127.0.0.1 \
    VISION_WORKER_PORT=8001 \
    VISION_MODEL_PATH=./backend/models_weights/yolo26n.pt \
    VISION_MODEL_DEVICE=cpu \
    VISION_MODEL_CLASSES=1,2,3,4,5,6,7,8 \
    VISION_INFERENCE_FPS=5 \
    POSTGRES_SSLMODE=require

EXPOSE 7860
CMD ["./start.sh"]
