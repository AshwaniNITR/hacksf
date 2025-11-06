# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and using stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_PROGRESS_BAR=on

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Upgrade pip (show progress)
RUN pip install --upgrade pip

# Install only CPU version of PyTorch and related libraries (with progress)
# RUN pip install \
#     torch==2.3.0+cpu \
#     torchvision==0.18.0+cpu \
#     torchaudio==2.3.0+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

# # Install facenet-pytorch and other core dependencies
# RUN pip install \
#     facenet-pytorch 

# Copy and install remaining requirements (optional)
COPY requirements.txt .
RUN pip install -r requirements.txt 

# Expose Flask port
EXPOSE 5000

# Command to run Flask app
CMD ["python", "app.py"]



