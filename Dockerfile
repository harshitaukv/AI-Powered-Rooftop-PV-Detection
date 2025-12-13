# Base image (lightweight, production-safe)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirement file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (runs inference script)
CMD ["python", "evaluate.py"]

https://hub.docker.com/repository/docker/harshitau/rooftop-pv-detector
Tags:
docker build -t harshitau/rooftop-pv-detector:v1.0 
docker tag harshitau//rooftop-pv-detector:latest
Push to DockerHub:
docker login
docker push harshitau/rooftop-pv-detector:v1.0
docker push harshitau/rooftop-pv-detector:latest



