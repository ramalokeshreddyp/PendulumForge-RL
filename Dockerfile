FROM python:3.11-slim

WORKDIR /app

# Runtime libraries required by pygame and rendering in containerized environments.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Install CPU-only PyTorch first (avoids downloading the 670 MB GPU wheel)
RUN pip install --no-cache-dir --retries 5 --timeout 180 \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 5 --timeout 180 -r requirements.txt

COPY . .

CMD ["python", "train.py", "--timesteps", "1000"]
