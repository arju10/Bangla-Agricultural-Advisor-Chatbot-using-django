# Use official Python 3.11 slim image (small size)
FROM python:3.11-slim

# Set environment variables to avoid Python buffering and ensure UTF-8 encoding
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LANG=C.UTF-8

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt /app/

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your project code
COPY . /app/

# Collect static files (if you use Django static files)
RUN python manage.py collectstatic --noinput

# Expose port 8000 (default Django port)
EXPOSE 8000

# Run the Django development server (for production, replace with gunicorn or other WSGI server)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
