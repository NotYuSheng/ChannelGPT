# Use the official Python image with a slim base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first for caching purposes
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app

# Copy the template config file to config.py
RUN cp config.template.py config.py

# Expose the FastAPI port
EXPOSE 8001

# Set the environment variable for uvicorn to run in production mode
ENV HOST 0.0.0.0
ENV PORT 8001

# Command to run the FastAPI app
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8001", "--log-level", "debug"]
