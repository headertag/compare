# Use a base image with Python and OpenCV pre-installed
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the project files into the container
COPY . /app/

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make the main script executable
RUN chmod +x main.py

# Command to run the application
CMD ["python", "main.py"]
