# Base image with PyTorch + Transformers
FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python packages
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all app files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default command to run the app
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.enableCORS=false"]
