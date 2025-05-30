FROM python:3.9-slim

# התקנת תלויות
RUN apt-get update && apt-get install -y \
    build-essential \  # הוסף את build-essential כדי להתקין g++
    libgl1-mesa-glx \
    cmake \
    libboost-all-dev \
    && pip install --no-cache-dir numpy opencv-python face-recognition requests

# העתקת הקוד שלך לתוך הקונטיינר
COPY . /app
WORKDIR /app

# הפעלת השרת
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

