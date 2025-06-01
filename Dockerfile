# ----------------------------------------
# Stage 1: Build environment – כעת נוודא שלא נבצע קומפילציה מיותרת של dlib
# ----------------------------------------
FROM python:3.11-slim AS builder

# 1. התקנת התלויות הריצה והבניה ההכרחיות (בלי boost מלא)
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

# 3. מתקינים dlib מה-wheel המוכן עבור Python 3.11 (ללא build מ־source)
#    הגרסה 19.22.99 ידועה ב־manylinux wheel ל־py3.11, כך ש‐pip לא ינסה לקומפל את dlib.
RUN pip install --no-cache-dir dlib==19.22.99

# 4. מתקינים face_recognition ללא תלות ב־dlib (כבר התקנו אותו בשורה הקודמת),
#    ואז כל שאר חבילות ה־Python
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

# 6. מעתיקים את כל חבילות ה־Python שהותקנו ב־builder (כולל dlib, face_recognition, וכו')
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 7. מעתיקים את קוד המקור שלכם (לדוגמה: main.py)
COPY . .

# 8. פותחים פורט 8000
EXPOSE 8000

# 9. פקודת ההפעלה
CMD ["uvicorn", "compare_face:app", "--host", "0.0.0.0", "--port", "8000"]