FROM python:3.11-slim

# Install NTP tools
RUN apt-get update && apt-get install -y ntpdate

# Sync time on startup (ignore errors if restricted)
RUN ntpdate -s time.nist.gov || true

# Copy app files
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
