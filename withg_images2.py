#
# import numpy as np
# import pickle
# import cv2
# from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
# from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# from sqlalchemy.orm import sessionmaker, relationship, Session
# from sqlalchemy.ext.declarative import declarative_base
# from contextlib import contextmanager
# from sqlalchemy.dialects.mysql import MEDIUMBLOB
# import face_recognition
# import matplotlib.pyplot as plt
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import StreamingResponse
# import io
# import base64
# from pydantic import BaseModel
#
# # הגדרת בסיס המודל
# Base = declarative_base()
#
# class Image(Base):
#     __tablename__ = 'images'
#     id = Column(Integer, primary_key=True)
#     image_data = Column(MEDIUMBLOB, nullable=False)
#     image_name = Column(String(255), nullable=False)
#     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
#
# class Face(Base):
#     __tablename__ = 'faces'
#     id = Column(Integer, primary_key=True)
#     encoding = Column(MEDIUMBLOB, nullable=False)
#     face_image_data = Column(MEDIUMBLOB, nullable=False)
#     images = relationship("ImageFaceLink", back_populates="face")
#
# class ImageFaceLink(Base):
#     __tablename__ = 'image_face_link'
#     id = Column(Integer, primary_key=True)
#     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
#     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
#     image = relationship("Image", back_populates="face_links")
#     face = relationship("Face", back_populates="images")
#
# # חיבור למסד הנתונים
# def create_database_connection():
#     try:
#         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image5")
#         Base.metadata.create_all(engine)
#         return sessionmaker(bind=engine)()
#     except Exception as e:
#         print(f"Error connecting to the database: {e}")
#         return None
#
# @contextmanager
# def get_session():
#     session = create_database_connection()
#     try:
#         yield session
#     finally:
#         session.close()
#
# # פונקציות לעיבוד תמונה
# def load_image(image_data):
#     print("start load_images")
#     nparr = np.frombuffer(image_data, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     print("load_image success")
#     return image
#
# def detect_faces(image):
#     print("detect_faces start")
#     temp=face_recognition.face_locations(image, model="hog")
#     print("detect_faces end")
#     return temp
#
#
# def extract_faces(image, face_locations):
#     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
#
# def display_faces(face_images):
#     print("display_faces start")
#
#     if not face_images:
#         print("No faces detected.")
#         return
#     plt.figure(figsize=(10, 10))
#     for i, face in enumerate(face_images):
#         plt.subplot(1, len(face_images), i + 1)
#         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#     plt.show()
#     print("display_faces end")
#
#
# def resize_image(image, scale_percent):
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
#
# # פונקציות לזיהוי פנים
# def recognize_faces(image, session, threshold=0.5):
#     print("recognize_faces start")
#     image_resized = resize_image(image, 20)
#     face_locations = detect_faces(image_resized)
#     recognized_faces_indices = []
#
#     for face_location in face_locations:
#         (top, right, bottom, left) = face_location
#         face_encoding = face_recognition.face_encodings(image_resized, [face_location])
#
#         if face_encoding:  # בדוק אם קידוד נמצא
#             face_encoding = face_encoding[0]
#             existing_face_id = face_exists(session, face_encoding, threshold)
#             recognized_faces_indices.append(existing_face_id)
#
#             if existing_face_id == -1:
#                 cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
#             else:
#                 cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
#                 cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             (255, 0, 0), 2)
#     print("recognize_faces end")
#
#     return image_resized, recognized_faces_indices
#
# def query_images_by_face_id(session, face_id):
#     print("get_images_by_face_id start")
#     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
#     images = []
#     for link in links:
#         image = session.query(Image).filter(Image.id == link.image_id).first()
#         if image:
#             images.append((image.id, image.image_name, image.image_data))
#     print("get_images_by_face_id end")
#     return images
#
# def display_images(images):
#     plt.figure(figsize=(10, 10))
#     for i, (image_id, image_name, image_data) in enumerate(images):
#         image_array = np.frombuffer(image_data, np.uint8)
#         image_decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         plt.subplot(1, len(images), i + 1)
#         plt.imshow(cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB))
#         plt.title(image_name)
#         plt.axis('off')
#     plt.show()
#
# # פונקציות למסד נתונים
# def get_all_faces_db(session):
#     print("get_all_faces start")
#     existing_faces = session.query(Face).all()
#     faces = []
#     for face in existing_faces:
#         try:
#             print("face.encoding"+type(face.encoding))
#             encoding = pickle.loads(face.encoding)
#             faces.append((face.id, encoding))
#         except Exception as e:
#             print(f"Error loading face encoding for ID {face.id}: {e}")
#     print("get_all_faces end")
#     return faces
#
# def face_exists(session, new_encoding, threshold=0.6):
#     print("face_exists start")
#     face_data = get_all_faces_db(session)
#     closest_id = -1
#     closest_distance = float('inf')
#
#     if not face_data:  # אם אין פנים, החזר -1
#         return closest_id
#     print(type(face_data))
#
#     for face_id, face_encoding in face_data:
#         print(type(face_id))
#         print(type(face_encoding))
#
#         print("existing_face_encoding = np.array(pickle.loads(face_encoding))")
#         existing_face_encoding = np.array(pickle.loads(face_encoding))  # המרת קידוד מ-pickle למערך NumPy
#         print("new_encoding = np.array(face_encoding)")
#         new_encoding = np.array(face_encoding)  # ודא שזה גם מערך NumPy
#         print("distance = np.linalg.norm(existing_face_encoding - new_encoding)")
#         distance = np.linalg.norm(existing_face_encoding - new_encoding)
#         print("after")
#         if distance < threshold and distance < closest_distance:
#             closest_distance = distance
#             closest_id = face_id
#     print("face_exists end")
#     return closest_id
#
# def image_exists(session, image_name):
#     return session.query(Image).filter(Image.image_name == image_name).first() is not None
#
# def save_image_to_db(session, image_name, image_data, face_encodings, face_images):
#     print("save_image_to_db start")
#     if image_exists(session, image_name):
#         print(f"Image '{image_name}' already exists in the database.")
#         print("save_image_to_db end")
#         return
#
#     new_image = Image(image_data=image_data, image_name=image_name)
#     session.add(new_image)
#     session.commit()
#
#     for encoding, face_image in zip(face_encodings, face_images):
#         existing_face_id = face_exists(session, encoding)
#
#         if existing_face_id == -1:
#             _, buffer = cv2.imencode('.jpg', face_image)
#             face_image_data = buffer.tobytes()
#             encoded_face = pickle.dumps(encoding)
#
#             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
#             session.add(new_face)
#             session.commit()
#             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
#             session.add(link)
#         else:
#             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
#             session.add(link)
#
#     session.commit()
#     print("Image and faces uploaded successfully.")
#     print("save_image_to_db end")
#
#
# # יצירת FastAPI
#
# app = FastAPI()
# #
#
# def get_db():
#     session = create_database_connection()
#     try:
#         yield session
#     finally:
#         session.close()
#
# @app.get("/images/")
# # def get_all_images(db: Session= Depends(get_db)):
# #     images = db.query(Image).all()
# #     result=[{
# #         "id": image.id,
# #         "name": image.image_name,
# #         "image_data": base64.b64encode(image.image_data).decode('utf-8')
# #     } for image in images]
# #     return result
# def get_all_images(db: Session= Depends(get_db)):
#     images = db.query(Image).all()
#     result=[{
#         "id": image_id,
#         "name": image_name,
#         "image_data": base64.b64encode(image_data).decode('utf-8')
#     } for image_id,image_name,image_data in images]
#     return result
# #
# # @app.get("/images/{image_id}")
# # def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
# #     image = db.query(Image).filter(Image.id == image_id).first()
# #     if not image:
# #         raise HTTPException(status_code=404, detail="Image not found")
# #     return {
# #         "id": image.id,
# #         "name": image.image_name,
# #         "image_data": f"data:image/jpeg;base64,{base64.b64encode(image.image_data).decode('utf-8')}"
# #     }
#
#
# @app.get("/images/{image_id}")
# def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
#     image = db.query(Image).filter(Image.id == image_id).first()
#     if not image:
#         raise HTTPException(status_code=404, detail="Image not found")
#
#     # הפוך את נתוני התמונה ל-stream
#     image_stream = io.BytesIO(image.image_data)
#     return StreamingResponse(image_stream, media_type="image/jpeg")
#
#
#
#
# @app.post("/images/")
# async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     if not file:
#         raise HTTPException(status_code=400, detail="No file provided")
#
#     try:
#         image_data = await file.read()
#         if not image_data:
#             raise HTTPException(status_code=400, detail="File is empty")
#
#         image = load_image(image_data)
#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image format")
#         face_locations = detect_faces(image)
#         if not face_locations:
#             raise HTTPException(status_code=400, detail="No faces detected in the image")
#
#         known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
#
#         # קח את השם מהקובץ
#         image_name = file.filename
#
#         save_image_to_db(db, image_name, image_data, known_faces, extract_faces(image, face_locations))
#         return {"detail": "Image uploaded successfully"}
#
#     except HTTPException as http_exception:
#         raise http_exception  # אם זו כבר HTTPException, פשוט להעלות אותה מחדש
#     except Exception as e:
#         print(f"Error: {e}")  # הדפסת השגיאה לקונסול
#         raise HTTPException(status_code=500, detail="Internal Server Error")
#
# @app.get("/faces/")
# def get_all_faces(db: Session = Depends(get_db)):
#     existing_faces = db.query(Face).all()
#     faces_data = []
#
#     for face in existing_faces:
#         # face_image_data = face.face_image_data  # הנח שהשדה הזה מכיל את נתוני התמונה
#         faces_data.append({
#             "id": face.id,
#             "image_data": base64.b64encode(face.face_image_data).decode('utf-8')  # החזרת נתוני התמונה
#         })
#
#     return faces_data
# #
# # @app.get("/faces/{face_id}")
# # def get_face_by_id(face_id: int, db: Session = Depends(get_db)):
# #     face = db.query(Face).filter(Face.id == face_id).first()
# #     if not face:
# #         raise HTTPException(status_code=404, detail="Face not found")
# #
# #     # החזרת מידע על הפנים כולל נתוני התמונה
# #     return {
# #         "id": face.id,
# #         "face_image_data": face.face_image_data,
# #         "encoding": face.encoding,
# #         "linked_images": [
# #             {
# #                 "id": link.image.id,
# #                 "image_name": link.image.image_name,
# #                 "image_data": link.image.image_data
# #             }
# #             for link in face.images
# #         ]
# #     }
# def aa():
#     print("aa")
# #
# # @app.get("/faces/{face_id}/images/")
# # def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
# #     # קבלת כל הקישורים בין תמונות לפנים
# #     # links = db.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# #     images = query_images_by_face_id(db, face_id)
# #     # החזרת רשימת התמונות הקשורות לפנים
# #     return [{
# #         "id": image_id,
# #         "image_name": image_name,
# #         "image_data": base64.b64encode(image_data).decode('utf-8')
# #     } for image_id,image_name,image_data in images]
#
#
# @app.get("/faces/{face_id}/images/")
# def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
#
#     images = query_images_by_face_id(db, face_id)
#     # החזרת רשימת התמונות הקשורות לפנים
#     res= [
#     StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")
#     for _,_,image_data in images]
#     return res
#
#
#
# #
# # app = FastAPI()
# #
# # # --- קונטרולר לשליפת כל התמונות ---
# # @app.get("/images/", response_model=List[ImageRead])
# # def get_all_images(db: Session = Depends(get_db)):
# #     return get_all_images_from_db(db)
#
# # --- קונטרולר לשליפת תמונות לפי face_id ---
# # @app.get("/faces/{face_id}/images/")
# # def get_images_for_face(face_id: int, db: Session = Depends(get_db)):
# #     images = query_images_by_face_id(db, face_id)
# #     if images is None:
# #         raise HTTPException(status_code=404, detail="Face not found or no images")
# #     return images
# # #
# # # --- קונטרולר להוספת תמונה חדשה ---
# # @app.post("/images/", response_model=ImageRead)
# # def add_image(image_data: ImageCreate, db: Session = Depends(get_db)):
# #     return create_new_image(db, image_data)
# # #
# # # --- קונטרולר למחיקת תמונה לפי ID ---
# # @app.delete("/images/{image_id}/")
# # def delete_image(image_id: int, db: Session = Depends(get_db)):
# #     deleted = delete_image_by_id(db, image_id)
# #     if not deleted:
# #         raise HTTPException(status_code=404, detail="Image not found")
# #     return {"detail": "Image deleted successfully"}
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8002)




