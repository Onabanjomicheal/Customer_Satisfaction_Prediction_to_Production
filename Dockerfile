FROM python:3.9-slim

WORKDIR /app

# Install system dependencies REQUIRED for xgboost, lightgbm, and scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip to handle modern wheel files
RUN pip install --no-cache-dir --upgrade pip 

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "app.py"]