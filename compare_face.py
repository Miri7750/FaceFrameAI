# Python 3.11+
import os
os.system('pip install dlib --no-cache-dir')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import numpy as np
import cv2
import face_recognition
import requests
import base64

app = FastAPI(title="Face Recognition Service")

class EncodingDto(BaseModel):
    id: int
    encoding: str  # Base64 של numpy.ndarray

class CompareIn(BaseModel):
    event_id: int
    image_url: HttpUrl
    existing_encodings: list[EncodingDto]
    threshold: float = 0.4

class MatchDto(BaseModel):
    top: int
    left: int
    width: int
    height: int
    matched_id: int | None
    distance: float

@app.post("/compare-faces/", response_model=list[MatchDto])
async def compare_faces(payload: CompareIn):
    # 1. הורדת התמונה
    resp = requests.get(payload.image_url)
    if resp.status_code != 200:
        raise HTTPException(400, "Cannot download image")
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)

    # 2. גילוי פנים וקידודים
    locs = face_recognition.face_locations(img, model="hog")
    encs = face_recognition.face_encodings(img, locs)

    # 3. פירוק existing encodings
    existing = [
        (e.id, np.frombuffer(base64.b64decode(e.encoding), dtype=np.float64))
        for e in payload.existing_encodings
    ]

    # 4. השוואה
    matches: list[MatchDto] = []
    for enc, (top, right, bottom, left) in zip(encs, locs):
        best_id, best_dist = None, float('inf')
        for eid, eenc in existing:
            d = np.linalg.norm(enc - eenc)
            if d < payload.threshold and d < best_dist:
                best_dist, best_id = d, eid

        matches.append(MatchDto(
            top=top,
            left=left,
            width=right-left,
            height=bottom-top,
            matched_id=best_id,
            distance=best_dist if best_id is not None else float('inf')
        ))
    return matches

# להרצה:
# uvicorn main:app --host 0.0.0.0 --port 8000
