# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . /app/

# Collect static files if needed
RUN python manage.py migrate

# Expose port 8000
EXPOSE 8000

# Run the Django development server (for production, replace with gunicorn or other WSGI server)
# CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Use Gunicorn as the WSGI server for production
CMD ["gunicorn", "bangla_agriculture_chatbot.wsgi:application", "--bind", "0.0.0.0:8000"]
