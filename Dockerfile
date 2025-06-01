# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# 1. התקנת ספריות ריצה מינימליות עבור OpenCV ו־dlib (מבלי לקמפל שום דבר)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. מעתיקים קובץ דרישות (אפשר להכין אותו במקביל לדוקר ישׂ)
COPY requirements.txt .

# 3. משדרגים pip ומתקינים את כל החבילות מתוך requirements.txt
#    שימו לב: גרסת dlib 19.22.1 מגיעה כ־manylinux wheel עבור Python 3.11,
#    ולכן לא תצטרכו לבנות אותה מ־source (וזה יחסוך זיכרון בזמן ה-build).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. מעתיקים את קוד היישום לתוך התמונה
COPY . .

# 5. פותחים את הפורט שה־FastAPI רץ עליו
EXPOSE 8000

# 6. פקודת הפעלה
CMD ["uvicorn", "compare_face:app", "--host", "0.0.0.0", "--port", "8000"]