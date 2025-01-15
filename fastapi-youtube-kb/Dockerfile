# Use the official Python image with a slim base
FROM python:3.10-slim

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for caching purposes
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Expose the FastAPI port
EXPOSE 8001

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug"]
