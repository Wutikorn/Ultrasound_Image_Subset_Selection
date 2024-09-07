# Base Image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY Keyframe_extractor.py /app/
COPY spectral_clustering.py /app/
COPY sononet /app/sononet/
COPY requirements.txt /app/
COPY best_ultrasound_resnet50.pth /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libgl1-mesa-glx \  
    && rm -rf /var/lib/apt/lists/*

# Install Python Requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run Keyframe extractor
CMD ["python", "Keyframe_extractor.py"]