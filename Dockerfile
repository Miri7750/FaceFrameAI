
# ----------------------------------------
# Stage 1: Build environment (ממנו מורידים ומתקינים את כל התלויות הכבדות)
# ----------------------------------------
FROM python:3.11-slim AS builder

# 1. מתקינים רק מה שדרוש כדי לבנות dlib/face_recognition/opencv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      pkg-config \
      python3-dev \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*
=======
# התקנת תלויות
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    cmake \
    libboost-all-dev \
    && pip install --no-cache-dir numpy opencv-python face-recognition requests
>>>>>>> origin/main

WORKDIR /app

# 2. מעלים pip ומתקינים את החבילות בפייתון (כולל dlib, face_recognition, opencv)
#    שימו לב: face-recognition תלוי ב-dlib, ואנחנו מנסים להביא בפיפ (wheel) במקום לבנות source
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      numpy \
      opencv-python \
      face-recognition \
      requests

# ----------------------------------------
# Stage 2: Runtime environment (ממנו בונים את התמונה הסופית, בלי build tools מיותרים)
# ----------------------------------------
FROM python:3.11-slim

# 3. מתקינים רק את ספריות הריצה של OpenCV (ולא את החבילות הכבדות לבנייה)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. מעתיקים את כל חבילות הפייתון שהותקנו בשלב ה-builder (מתוך תיקיית site-packages)
#    ובכלים שהותקנו בלוקל (כגון uvicorn)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 5. מעתיקים את קוד המקור שלכם (main.py, ייתכן ששמו שונה ל־main.py או app.py לפי הפרויקט)
COPY . .

# 6. פותחים את הפורט שעליו השירות ירוץ
EXPOSE 8000

# 7. הפקודה הראשית להרצה
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
