# ----------------------------------------
# Stage 1: Build environment – מתקינים את dlib מ-wheel נכון
# ----------------------------------------
FROM python:3.11-slim AS builder

# 1. התקנת התלויות הדרושות להרכבת ספריות פייתון כבדות (ללא boost מלא)
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

WORKDIR /app

# 2. מעלים pip לגרסה עדכנית
RUN pip install --upgrade pip

# 3. מתקינים dlib מ-wheel עבור Python 3.11 (גרסה שיש לה תמיכה ב-manylinux)
RUN pip install --no-cache-dir dlib==19.24.1

# 4. מתקינים face_recognition (ללא תלות ב-dlib), ושאר חבילות ה-Python
RUN pip install --no-cache-dir \
      face-recognition --no-deps \
      fastapi \
      "uvicorn[standard]" \
      numpy \
      opencv-python \
      requests

# ----------------------------------------
# Stage 2: Runtime מינימלי
# ----------------------------------------
FROM python:3.11-slim

# 5. התקנת ספריות ריצה של OpenCV בלבד
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 6. מעתיקים את חבילות ה-Python שהותקנו בשלב ה-builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 7. מעתיקים את קוד המקור שלכם
COPY . .

# 8. פותחים פורט 8000
EXPOSE 8000

# 9. פקודת ההפעלה
CMD ["uvicorn", "compare_face:app", "--host", "0.0.0.0", "--port", "8000"]