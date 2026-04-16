FROM python:3.12-slim

WORKDIR /app

# Install system deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Download model files (turn detector, silero VAD)
RUN python main.py download-files

CMD ["python", "main.py", "dev"]
