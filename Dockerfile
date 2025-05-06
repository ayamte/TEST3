FROM python:3.10  
  
# Install system dependencies  
RUN apt-get update && apt-get install -y poppler-utils tesseract-ocr  
  
# Set working directory  
WORKDIR /app  
  
# Copy requirements file  
COPY requirements.txt .  
  
# Install Python dependencies  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Copy application code  
COPY . .  
  
# Create necessary directories  
RUN mkdir -p data/cvs data/outputs data/profiles  
  
# Expose port for Flask  
EXPOSE 5000  
  
# Set environment variables  
ENV PYTHONPATH=/app  
ENV FLASK_APP=app.py  
  
# Command to run the application  
CMD ["flask", "run", "--host=0.0.0.0"]