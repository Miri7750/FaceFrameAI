# # import pickle
# # import cv2
# # import numpy as np
# # import face_recognition
# # import matplotlib.pyplot as plt
# # from sqlalchemy.dialects.mysql import LONGBLOB
# # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # from contextlib import contextmanager
# # from fastapi import FastAPI, UploadFile, File, Depends
# # from sqlalchemy.orm import Session
# #
# # # הגדרת בסיס המודל
# # Base = declarative_base()
# #
# #
# # class Image(Base):
# #     __tablename__ = 'images'
# #     id = Column(Integer, primary_key=True)
# #     image_data = Column(LONGBLOB, nullable=False)
# #     image_name = Column(String(255), nullable=False)
# #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# #
# #
# # class Face(Base):
# #     __tablename__ = 'faces'
# #     id = Column(Integer, primary_key=True)
# #     encoding = Column(LONGBLOB, nullable=False)
# #     face_image_data = Column(LONGBLOB, nullable=False)
# #     images = relationship("ImageFaceLink", back_populates="face")
# #
# #
# # class ImageFaceLink(Base):
# #     __tablename__ = 'image_face_link'
# #     id = Column(Integer, primary_key=True)
# #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# #     image = relationship("Image", back_populates="face_links")
# #     face = relationship("Face", back_populates="images")
# #
# #
# # # חיבור למסד הנתונים
# # def create_database_connection():
# #     engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# #     Base.metadata.create_all(engine)
# #     return sessionmaker(bind=engine)()
# #
# #
# # @contextmanager
# # def get_session():
# #     session = create_database_connection()
# #     try:
# #         yield session
# #     finally:
# #         session.close()
# #
# #
# # # פונקציות לעיבוד תמונה
# # def detect_faces(image):
# #     return face_recognition.face_locations(image, model="hog")
# #
# #
# # def extract_faces(image, face_locations):
# #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# #
# #
# # # פונקציות לזיהוי פנים
# # def recognize_faces(image, session, known_faces, threshold=0.5):
# #     image_resized = cv2.resize(image, (0, 0), fx=0.2, fy=0.2)
# #     face_locations = detect_faces(image_resized)
# #     recognized_faces_indices = []
# #
# #     for face_location in face_locations:
# #         (top, right, bottom, left) = face_location
# #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# #         existing_face_id = face_exists(session, face_encoding, threshold)
# #         recognized_faces_indices.append(existing_face_id)
# #
# #         if existing_face_id == -1:
# #             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)
# #         else:
# #             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)
# #
# #     return image_resized, recognized_faces_indices
# #
# #
# # def get_images_by_face_id(session, face_id):
# #     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# #     images = []
# #     for link in links:
# #         image = session.query(Image).filter(Image.id == link.image_id).first()
# #         if image:
# #             images.append((image.id, image.image_name, image.image_data))
# #     return images
# #
# #
# # def face_exists(session, new_encoding, threshold=0.6):
# #     existing_faces = session.query(Face).all()
# #     closest_id = -1
# #     closest_distance = float('inf')
# #
# #     for face in existing_faces:
# #         distance = face_recognition.face_distance([pickle.loads(face.encoding)], new_encoding)[0]
# #         if distance < threshold and distance < closest_distance:
# #             closest_distance = distance
# #             closest_id = face.id
# #
# #     return closest_id
# #
# #
# # def image_exists(session, image_name):
# #     return session.query(Image).filter(Image.image_name == image_name).first() is not None
# #
# #
# # async def save_image_to_db(session, image_file, face_encodings, face_images):
# #     image_data = await image_file.read()
# #     image_name = image_file.filename
# #
# #     if image_exists(session, image_name):
# #         return {"message": f"Image '{image_name}' already exists in the database."}
# #
# #     new_image = Image(image_data=image_data, image_name=image_name)
# #     session.add(new_image)
# #     session.commit()
# #
# #     for encoding, face_image in zip(face_encodings, face_images):
# #         existing_face_id = face_exists(session, encoding)
# #
# #         if existing_face_id == -1:
# #             _, buffer = cv2.imencode('.jpg', face_image)
# #             face_image_data = buffer.tobytes()
# #             encoded_face = pickle.dumps(encoding)
# #
# #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# #             session.add(new_face)
# #             session.commit()
# #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# #             session.add(link)
# #         else:
# #             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
# #             session.add(link)
# #
# #     session.commit()
# #     return {"message": "Image and faces uploaded successfully."}
# #
# #
# # app = FastAPI()
# #
# #
# # def get_db():
# #     db = create_database_connection()
# #     try:
# #         yield db
# #     finally:
# #         db.close()
# #
# #
# # @app.post("/upload-image/")
# # async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# #     face_encodings = []  # הכנס את הקידודים הידועים שלך
# #     face_images = []  # הכנס את התמונות של הפנים שלך
# #     return await save_image_to_db(db, file, face_encodings, face_images)
# #
# #
# # @app.post("/recognize-faces/")
# # async def recognize_faces_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
# #     image_data = await file.read()
# #     image_array = np.frombuffer(image_data, np.uint8)
# #     image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# #
# #     known_faces = []  # הכנס את הקידודים הידועים שלך
# #     recognized_image, recognized_faces_indices = recognize_faces(image, db, known_faces)
# #
# #     return {"recognized_faces": recognized_faces_indices}
# #
# #
# # @app.get("/images/{face_id}")
# # def get_images(face_id: int, db: Session = Depends(get_db)):
# #     images = get_images_by_face_id(db, face_id)
# #     return {"images": images}
# #
# #
# # if __name__ == "__main__":
# #     import uvicorn
# #
# #     uvicorn.run(app, host="127.0.0.1", port=8000)
# import pickle
# import cv2
# import numpy as np
# import face_recognition
# from sqlalchemy.dialects.mysql import LONGBLOB
# from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# from contextlib import contextmanager
# from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
# from sqlalchemy.orm import Session
#
# # הגדרת בסיס המודל
# Base = declarative_base()
#
#
# class Image(Base):
#     __tablename__ = 'images'
#     id = Column(Integer, primary_key=True)
#     image_data = Column(LONGBLOB, nullable=False)
#     image_name = Column(String(255), nullable=False)
#     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
#
#
# class Face(Base):
#     __tablename__ = 'faces'
#     id = Column(Integer, primary_key=True)
#     encoding = Column(LONGBLOB, nullable=False)
#     face_image_data = Column(LONGBLOB, nullable=False)
#     images = relationship("ImageFaceLink", back_populates="face")
#
#
# class ImageFaceLink(Base):
#     __tablename__ = 'image_face_link'
#     id = Column(Integer, primary_key=True)
#     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
#     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
#     image = relationship("Image", back_populates="face_links")
#     face = relationship("Face", back_populates="images")
#
#
# # חיבור למסד הנתונים
# def create_database_connection():
#     engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
#     Base.metadata.create_all(engine)
#     return sessionmaker(bind=engine)()
#
#
# @contextmanager
# def get_session():
#     session = create_database_connection()
#     try:
#         yield session
#     finally:
#         session.close()
#
#
# def detect_faces(image):
#     return face_recognition.face_locations(image, model="hog")
#
#
# def extract_faces(image, face_locations):
#     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
#
#
# async def save_image_to_db(session, image_file):
#     image_data = await image_file.read()
#     image_name = image_file.filename
#
#     if image_exists(session, image_name):
#         raise HTTPException(status_code=400, detail=f"Image '{image_name}' already exists in the database.")
#
#     new_image = Image(image_data=image_data, image_name=image_name)
#     session.add(new_image)
#     session.commit()
#     return new_image.id
#
#
# def face_exists(session, new_encoding, threshold=0.6):
#     existing_faces = session.query(Face).all()
#     closest_id = -1
#     closest_distance = float('inf')
#
#     for face in existing_faces:
#         distance = face_recognition.face_distance([pickle.loads(face.encoding)], new_encoding)[0]
#         if distance < threshold and distance < closest_distance:
#             closest_distance = distance
#             closest_id = face.id
#
#     return closest_id
#
#
# async def save_faces_to_db(session, image_id, face_encodings, face_images):
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
#             link = ImageFaceLink(image_id=image_id, face_id=new_face.id)
#             session.add(link)
#         else:
#             link = ImageFaceLink(image_id=image_id, face_id=existing_face_id)
#             session.add(link)
#
#     session.commit()
#
#
# def image_exists(session, image_name):
#     return session.query(Image).filter(Image.image_name == image_name).first() is not None
#
#
# app = FastAPI()
#
#
# def get_db():
#     db = create_database_connection()
#     try:
#         yield db
#     finally:
#         db.close()
#
# #
# # @app.post("/upload-image/")
# # async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# #     try:
# #         image_id = await save_image_to_db(db, file)
# #
# #         # עיבוד התמונה לזיהוי פנים
# #         image_data = await file.read()
# #         image_array = np.frombuffer(image_data, np.uint8)
# #         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# #         face_locations = detect_faces(image)
# #         face_images = extract_faces(image, face_locations)
# #         face_encodings = [face_recognition.face_encodings(image, [loc])[0] for loc in face_locations]
# #
# #         await save_faces_to_db(db, image_id, face_encodings, face_images)
# #
# #         return {"message": "Image and faces uploaded successfully.", "image_id": image_id}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
#
# @app.post("/upload-image/")
# async def upload_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     try:
#         image_data = await file.read()
#         if not image_data:
#             raise HTTPException(status_code=400, detail="No data read from the file.")
#
#         image_id = await save_image_to_db(db, file)
#
#         # עיבוד התמונה לזיהוי פנים
#         image_array = np.frombuffer(image_data, np.uint8)
#         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         if image is None:
#             raise HTTPException(status_code=400, detail="Failed to decode image.")
#
#         face_locations = detect_faces(image)
#         face_images = extract_faces(image, face_locations)
#         face_encodings = [face_recognition.face_encodings(image, [loc])[0] for loc in face_locations]
#
#         await save_faces_to_db(db, image_id, face_encodings, face_images)
#
#         return {"message": "Image and faces uploaded successfully.", "image_id": image_id}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# #
# # @app.get("/images/{face_id}")
# # def get_images(face_id: int, db: Session = Depends(get_db)):
# #     try:
# #         links = db.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# #         images = []
# #         for link in links:
# #             image = db.query(Image).filter(Image.id == link.image_id).first()
# #             if image:
# #                 images.append((image.id, image.image_name, image.image_data))
# #         return {"images": images}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))
# import base64
#
# @app.get("/images/{face_id}")
# def get_images(face_id: int, db: Session = Depends(get_db)):
#     try:
#         links = db.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
#         images = []
#         for link in links:
#             image = db.query(Image).filter(Image.id == link.image_id).first()
#             if image:
#                 # המרת הנתונים לפורמט Base64
#                 image_base64 = base64.b64encode(image.image_data).decode('utf-8')
#                 images.append({
#                     "id": image.id,
#                     "name": image.image_name,
#                     "data": image_base64
#                 })
#         return {"images": images}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="127.0.0.1", port=8000)
