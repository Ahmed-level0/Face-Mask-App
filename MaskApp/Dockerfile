# Use stable Python (NOT 3.13)
FROM python:3.12-slim

# Prevent Python from writing pyc files & buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (IMPORTANT for opencv + ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Collect static files (if using Django static)
RUN python manage.py collectstatic --noinput || true

# Expose port (Railway uses $PORT)
EXPOSE 8000

# Run with gunicorn
CMD ["sh", "-c", "gunicorn MaskApp.wsgi:application --bind 0.0.0.0:$PORT"]
