FROM python:3.9-slim

# התקנת תלויות
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && pip install --no-cache-dir fastapi uvicorn numpy opencv-python face-recognition requests pydantic

# העתקת הקוד שלך לתוך הקונטיינר
COPY . /app
WORKDIR /app

# הפעלת השרת
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
