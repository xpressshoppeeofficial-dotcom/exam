FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLOv8 model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Copy application code
COPY app.py .
COPY templates/ templates/

# Create temp directory for uploads
RUN mkdir -p /tmp/uploads

# Expose port
EXPOSE 8080

# Run the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "app:app"]
