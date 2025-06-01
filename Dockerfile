## ========================= Stage 1: Builder =========================
FROM python:3.11-slim AS builder

# 1. מגדירים את MAKEFLAGS כך ש-CMake ו-make יריצו קומפילציה בליבה אחת בלבד
#    (בכך מורידים משמעותית את צריכת הזיכרון בעת בניית dlib)
ENV MAKEFLAGS="-j1"

# 2. מתקינים כלי פיתוח ו-CMake הדרושים לבניית dlib ו-face-recognition-models
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

# 3. מעתיקים את קובץ ה־requirements החוצה
COPY requirements.txt .

# 4. יוצרים גלגלים (wheels) לכל התלויות כולל dlib ו־face-recognition-models,
#    בצורה אוטונומית בתוך התיקייה /wheels
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels


# ========================= Stage 2: Runtime =========================
FROM python:3.11-slim

# 5. מתקינים רק את הספריות הדרושות בזמן ריצה (ללא כל כלי הפיתוח)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 6. מעתיקים את כל הגלגלים משלב ה-builder ומתקינים אותם ללא בנייה חוזרת
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# 7. מתקינים את כל התלויות מתוך גלגלים בלבד (ללא גישה לרשת)
RUN pip install --no-index --find-links /wheels -r requirements.txt

# 8. מעתיקים את קוד היישום
COPY . .

# 9. מגדירים את פקודת ההפעלה
CMD ["uvicorn", "compare_face:app", "--host", "0.0.0.0", "--port", "8000"]