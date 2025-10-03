# Use official Python image (slim for fast build)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies (caches this layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]