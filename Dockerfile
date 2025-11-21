# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variable for Python
ENV PYTHONUNBUFFERED=1

# Default command when container runs
CMD ["bash"]
