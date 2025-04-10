# # # # # # import pickle
# # # # # # import cv2
# # # # # # import numpy as np
# # # # # # import face_recognition
# # # # # # import matplotlib.pyplot as plt
# # # # # # from sqlalchemy.dialects.mysql import MEDIUMBLOB
# # # # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # # # # # from contextlib import contextmanager
# # # # # # from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
# # # # # # from fastapi.responses import StreamingResponse
# # # # # # import io
# # # # # #
# # # # # # # הגדרת בסיס המודל
# # # # # # Base = declarative_base()
# # # # # #
# # # # # #
# # # # # # class Image(Base):
# # # # # #     __tablename__ = 'images'
# # # # # #     id = Column(Integer, primary_key=True)
# # # # # #     image_data = Column(MEDIUMBLOB, nullable=False)
# # # # # #     image_name = Column(String(255), nullable=False)
# # # # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # # # #
# # # # # #
# # # # # # class Face(Base):
# # # # # #     __tablename__ = 'faces'
# # # # # #     id = Column(Integer, primary_key=True)
# # # # # #     encoding = Column(MEDIUMBLOB, nullable=False)
# # # # # #     face_image_data = Column(MEDIUMBLOB, nullable=False)
# # # # # #     images = relationship("ImageFaceLink", back_populates="face")
# # # # # #
# # # # # #
# # # # # # class ImageFaceLink(Base):
# # # # # #     __tablename__ = 'image_face_link'
# # # # # #     id = Column(Integer, primary_key=True)
# # # # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # # # #     image = relationship("Image", back_populates="face_links")
# # # # # #     face = relationship("Face", back_populates="images")
# # # # # #
# # # # # #
# # # # # # # חיבור למסד הנתונים
# # # # # # def create_database_connection():
# # # # # #     try:
# # # # # #         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# # # # # #         Base.metadata.create_all(engine)
# # # # # #         return sessionmaker(bind=engine)()
# # # # # #     except Exception as e:
# # # # # #         print(f"Error connecting to the database: {e}")
# # # # # #         return None
# # # # # #
# # # # # #
# # # # # # @contextmanager
# # # # # # def get_session():
# # # # # #     session = create_database_connection()
# # # # # #     try:
# # # # # #         yield session
# # # # # #     finally:
# # # # # #         session.close()
# # # # # #
# # # # # #
# # # # # # # פונקציות לעיבוד תמונה
# # # # # # def load_image(image_path):
# # # # # #     image = cv2.imread(image_path)
# # # # # #     if image is None:
# # # # # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # # # # #     return image
# # # # # #
# # # # # #
# # # # # # def detect_faces(image):
# # # # # #     return face_recognition.face_locations(image, model="hog")
# # # # # #
# # # # # #
# # # # # # def extract_faces(image, face_locations):
# # # # # #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# # # # # #
# # # # # #
# # # # # # def display_faces(face_images):
# # # # # #     if not face_images:
# # # # # #         print("No faces detected.")
# # # # # #         return
# # # # # #     plt.figure(figsize=(10, 10))
# # # # # #     for i, face in enumerate(face_images):
# # # # # #         plt.subplot(1, len(face_images), i + 1)
# # # # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # # # #         plt.axis('off')
# # # # # #     plt.show()
# # # # # #
# # # # # #
# # # # # # def resize_image(image, scale_percent):
# # # # # #     width = int(image.shape[1] * scale_percent / 100)
# # # # # #     height = int(image.shape[0] * scale_percent / 100)
# # # # # #     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
# # # # # #
# # # # # #
# # # # # # # פונקציות לזיהוי פנים
# # # # # # def recognize_faces(image, session, known_faces, threshold=0.5):
# # # # # #     image_resized = resize_image(image, 20)
# # # # # #     face_locations = detect_faces(image_resized)
# # # # # #     recognized_faces_indices = []
# # # # # #
# # # # # #     for face_location in face_locations:
# # # # # #         (top, right, bottom, left) = face_location
# # # # # #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# # # # # #
# # # # # #         existing_face_id = face_exists(session, face_encoding, threshold)
# # # # # #         recognized_faces_indices.append(existing_face_id)
# # # # # #
# # # # # #         if existing_face_id == -1:
# # # # # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
# # # # # #         else:
# # # # # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
# # # # # #             cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# # # # # #                         (255, 0, 0), 2)
# # # # # #
# # # # # #     return image_resized, recognized_faces_indices
# # # # # #
# # # # # #
# # # # # # def get_images_by_face_id(session, face_id):
# # # # # #     """Retrieve all images associated with a specific face_id."""
# # # # # #     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# # # # # #     images = []
# # # # # #     for link in links:
# # # # # #         image = session.query(Image).filter(Image.id == link.image_id).first()
# # # # # #         if image:
# # # # # #             images.append((image.id, image.image_name, image.image_data))
# # # # # #     return images
# # # # # #
# # # # # #
# # # # # # def display_images(images):
# # # # # #     """Display images given a list of image tuples (id, name, data)."""
# # # # # #     plt.figure(figsize=(10, 10))
# # # # # #     for i, (image_id, image_name, image_data) in enumerate(images):
# # # # # #         image_array = np.frombuffer(image_data, np.uint8)
# # # # # #         image_decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# # # # # #         plt.subplot(1, len(images), i + 1)
# # # # # #         plt.imshow(cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB))
# # # # # #         plt.title(image_name)
# # # # # #         plt.axis('off')
# # # # # #     plt.show()
# # # # # #
# # # # # #
# # # # # # # פונקציות למסד נתונים
# # # # # # def get_all_faces(session):
# # # # # #     existing_faces = session.query(Face).all()
# # # # # #     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
# # # # # #
# # # # # #
# # # # # # def face_exists(session, new_encoding, threshold=0.6):
# # # # # #     face_data = get_all_faces(session)
# # # # # #     closest_id = -1
# # # # # #     closest_distance = float('inf')
# # # # # #
# # # # # #     for face_id, face_encoding in face_data:
# # # # # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # # # # #         if distance < threshold and distance < closest_distance:
# # # # # #             closest_distance = distance
# # # # # #             closest_id = face_id
# # # # # #
# # # # # #     return closest_id
# # # # # #
# # # # # #
# # # # # # def image_exists(session, image_name):
# # # # # #     """Check if an image with the given name already exists in the database."""
# # # # # #     return session.query(Image).filter(Image.image_name == image_name).first() is not None
# # # # # #
# # # # # #
# # # # # # def save_image_to_db(session, image_path, face_encodings, face_images):
# # # # # #     image_name = image_path.split('/')[-1]
# # # # # #
# # # # # #     if image_exists(session, image_name):
# # # # # #         print(f"Image '{image_name}' already exists in the database.")
# # # # # #         return
# # # # # #
# # # # # #     with open(image_path, 'rb') as file:
# # # # # #         image_data = file.read()
# # # # # #
# # # # # #     new_image = Image(image_data=image_data, image_name=image_name)
# # # # # #     session.add(new_image)
# # # # # #     session.commit()
# # # # # #
# # # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # # #         existing_face_id = face_exists(session, encoding)
# # # # # #
# # # # # #         if existing_face_id == -1:
# # # # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # # # #             face_image_data = buffer.tobytes()
# # # # # #             encoded_face = pickle.dumps(encoding)
# # # # # #
# # # # # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # # # # #             session.add(new_face)
# # # # # #             session.commit()
# # # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # # #             session.add(link)
# # # # # #         else:
# # # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
# # # # # #             session.add(link)
# # # # # #
# # # # # #     session.commit()
# # # # # #     print("Image and faces uploaded successfully.")
# # # # # #
# # # # # #
# # # # # # # יצירת FastAPI
# # # # # # app = FastAPI()
# # # # # #
# # # # # #
# # # # # # # חיבור למסד הנתונים
# # # # # # def get_db():
# # # # # #     session = create_database_connection()
# # # # # #     try:
# # # # # #         yield session
# # # # # #     finally:
# # # # # #         session.close()
# # # # # #
# # # # # #
# # # # # # @app.get("/images/")
# # # # # # def get_all_images(db: Session = Depends(get_db)):
# # # # # #     images = db.query(Image).all()
# # # # # #     return [{"id": image.id, "name": image.image_name} for image in images]
# # # # # #
# # # # # #
# # # # # # @app.get("/images/{image_id}")
# # # # # # def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
# # # # # #     image = db.query(Image).filter(Image.id == image_id).first()
# # # # # #     if not image:
# # # # # #         raise HTTPException(status_code=404, detail="Image not found")
# # # # # #     return {"id": image.id, "name": image.image_name}
# # # # # #
# # # # # #
# # # # # # @app.post("/images/")
# # # # # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # # # # #     image_path = f"/path/to/save/{file.filename}"  # Update this path
# # # # # #     with open(image_path, "wb") as buffer:
# # # # # #         buffer.write(await file.read())
# # # # # #
# # # # # #     image = load_image(image_path)
# # # # # #     face_locations = detect_faces(image)
# # # # # #     face_images = extract_faces(image, face_locations)
# # # # # #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # # # # #
# # # # # #     save_image_to_db(db, image_path, known_faces, face_images)
# # # # # #     return {"detail": "Image uploaded successfully"}
# # # # # #
# # # # # #
# # # # # # @app.get("/faces/")
# # # # # # def get_all_faces(db: Session = Depends(get_db)):
# # # # # #     existing_faces = db.query(Face).all()
# # # # # #     return [{"id": face.id} for face in existing_faces]
# # # # # #
# # # # # #
# # # # # # @app.get("/faces/{face_id}")
# # # # # # def get_face_by_id(face_id: int, db: Session = Depends(get_db)):
# # # # # #     face = db.query(Face).filter(Face.id == face_id).first()
# # # # # #     if not face:
# # # # # #         raise HTTPException(status_code=404, detail="Face not found")
# # # # # #     return {"id": face.id}
# # # # # #
# # # # # #
# # # # # # @app.get("/faces/{face_id}/images/")
# # # # # # def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
# # # # # #     images = get_images_by_face_id(db, face_id)
# # # # # #     return [{"id": image[0], "name": image[1]} for image in images]
# # # # # #
# # # # # #
# # # # # # if __name__ == "__main__":
# # # # # #     import uvicorn
# # # # # #
# # # # # #     uvicorn.run(app, host="127.0.0.1", port=8000)
# # # # # import pickle
# # # # # import cv2
# # # # # import numpy as np
# # # # # import face_recognition
# # # # # import matplotlib.pyplot as plt
# # # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session  # הוספת Session כאן
# # # # # from sqlalchemy.dialects.mysql import MEDIUMBLOB
# # # # # from contextlib import contextmanager
# # # # # from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
# # # # # import io
# # # # #
# # # # # # הגדרת בסיס המודל
# # # # # Base = declarative_base()
# # # # #
# # # # #
# # # # # class Image(Base):
# # # # #     __tablename__ = 'images'
# # # # #     id = Column(Integer, primary_key=True)
# # # # #     image_data = Column(MEDIUMBLOB, nullable=False)
# # # # #     image_name = Column(String(255), nullable=False)
# # # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # # #
# # # # #
# # # # # class Face(Base):
# # # # #     __tablename__ = 'faces'
# # # # #     id = Column(Integer, primary_key=True)
# # # # #     encoding = Column(MEDIUMBLOB, nullable=False)
# # # # #     face_image_data = Column(MEDIUMBLOB, nullable=False)
# # # # #     images = relationship("ImageFaceLink", back_populates="face")
# # # # #
# # # # #
# # # # # class ImageFaceLink(Base):
# # # # #     __tablename__ = 'image_face_link'
# # # # #     id = Column(Integer, primary_key=True)
# # # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # # #     image = relationship("Image", back_populates="face_links")
# # # # #     face = relationship("Face", back_populates="images")
# # # # #
# # # # #
# # # # # # חיבור למסד הנתונים
# # # # # def create_database_connection():
# # # # #     try:
# # # # #         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image4")
# # # # #         Base.metadata.create_all(engine)
# # # # #         return sessionmaker(bind=engine)()
# # # # #     except Exception as e:
# # # # #         print(f"Error connecting to the database: {e}")
# # # # #         return None
# # # # #
# # # # #
# # # # # @contextmanager
# # # # # def get_session():
# # # # #     session = create_database_connection()
# # # # #     try:
# # # # #         yield session
# # # # #     finally:
# # # # #         session.close()
# # # # #
# # # # #
# # # # # # פונקציות לעיבוד תמונה
# # # # #
# # # # #
# # # # # # def load_image(image_path):
# # # # # #     image = cv2.imread(image_path)
# # # # # #     if image is None:
# # # # # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # # # # #     return image
# # # # #
# # # # # def load_image(image_data):
# # # # #     # המרת ה-BLOB למערך NumPy
# # # # #     nparr = np.frombuffer(image_data, np.uint8)
# # # # #     # קריאת התמונה בעזרת OpenCV
# # # # #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# # # # #     return image
# # # # #
# # # # #
# # # # # def detect_faces(image):
# # # # #     return face_recognition.face_locations(image, model="hog")
# # # # #
# # # # #
# # # # # def extract_faces(image, face_locations):
# # # # #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# # # # #
# # # # #
# # # # # def display_faces(face_images):
# # # # #     if not face_images:
# # # # #         print("No faces detected.")
# # # # #         return
# # # # #     plt.figure(figsize=(10, 10))
# # # # #     for i, face in enumerate(face_images):
# # # # #         plt.subplot(1, len(face_images), i + 1)
# # # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # # #         plt.axis('off')
# # # # #     plt.show()
# # # # #
# # # # #
# # # # # def resize_image(image, scale_percent):
# # # # #     width = int(image.shape[1] * scale_percent / 100)
# # # # #     height = int(image.shape[0] * scale_percent / 100)
# # # # #     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
# # # # #
# # # # #
# # # # # # פונקציות לזיהוי פנים
# # # # # def recognize_faces(image, session, known_faces, threshold=0.5):
# # # # #     image_resized = resize_image(image, 20)
# # # # #     face_locations = detect_faces(image_resized)
# # # # #     recognized_faces_indices = []
# # # # #
# # # # #     for face_location in face_locations:
# # # # #         (top, right, bottom, left) = face_location
# # # # #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# # # # #
# # # # #         existing_face_id = face_exists(session, face_encoding, threshold)
# # # # #         recognized_faces_indices.append(existing_face_id)
# # # # #
# # # # #         if existing_face_id == -1:
# # # # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
# # # # #         else:
# # # # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
# # # # #             cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# # # # #                         (255, 0, 0), 2)
# # # # #
# # # # #     return image_resized, recognized_faces_indices
# # # # #
# # # # #
# # # # # def get_images_by_face_id(session, face_id):
# # # # #     """Retrieve all images associated with a specific face_id."""
# # # # #     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# # # # #     images = []
# # # # #     for link in links:
# # # # #         image = session.query(Image).filter(Image.id == link.image_id).first()
# # # # #         if image:
# # # # #             images.append((image.id, image.image_name, image.image_data))
# # # # #     return images
# # # # #
# # # # #
# # # # # def display_images(images):
# # # # #     """Display images given a list of image tuples (id, name, data)."""
# # # # #     plt.figure(figsize=(10, 10))
# # # # #     for i, (image_id, image_name, image_data) in enumerate(images):
# # # # #         image_array = np.frombuffer(image_data, np.uint8)
# # # # #         image_decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# # # # #         plt.subplot(1, len(images), i + 1)
# # # # #         plt.imshow(cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB))
# # # # #         plt.title(image_name)
# # # # #         plt.axis('off')
# # # # #     plt.show()
# # # # #
# # # # #
# # # # # # פונקציות למסד נתונים
# # # # # def get_all_faces(session):
# # # # #     existing_faces = session.query(Face).all()
# # # # #     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
# # # # #
# # # # #
# # # # # def face_exists(session, new_encoding, threshold=0.6):
# # # # #     face_data = get_all_faces(session)
# # # # #     closest_id = -1
# # # # #     closest_distance = float('inf')
# # # # #
# # # # #     for face_id, face_encoding in face_data:
# # # # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # # # #         if distance < threshold and distance < closest_distance:
# # # # #             closest_distance = distance
# # # # #             closest_id = face_id
# # # # #
# # # # #     return closest_id
# # # # #
# # # # #
# # # # # def image_exists(session, image_name):
# # # # #     """Check if an image with the given name already exists in the database."""
# # # # #     return session.query(Image).filter(Image.image_name == image_name).first() is not None
# # # # #
# # # # # #
# # # # # # def save_image_to_db(session, image_path, face_encodings, face_images):
# # # # # #     image_name = image_path.split('/')[-1]
# # # # # #
# # # # # #     if image_exists(session, image_name):
# # # # # #         print(f"Image '{image_name}' already exists in the database.")
# # # # # #         return
# # # # # #
# # # # # #     with open(image_path, 'rb') as file:
# # # # # #         image_data = file.read()
# # # # # #
# # # # # #     new_image = Image(image_data=image_data, image_name=image_name)
# # # # # #     session.add(new_image)
# # # # # #     session.commit()
# # # # # #
# # # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # # #         existing_face_id = face_exists(session, encoding)
# # # # # #
# # # # # #         if existing_face_id == -1:
# # # # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # # # #             face_image_data = buffer.tobytes()
# # # # # #             encoded_face = pickle.dumps(encoding)
# # # # # #
# # # # # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # # # # #             session.add(new_face)
# # # # # #             session.commit()
# # # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # # #             session.add(link)
# # # # # #         else:
# # # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
# # # # # #             session.add(link)
# # # # # #
# # # # # #     session.commit()
# # # # # #     print("Image and faces uploaded successfully.")
# # # # #
# # # # # def save_image_to_db(session, image_path, face_encodings, face_images):
# # # # #     image_name = image_path.split('/')[-1]
# # # # #
# # # # #     if image_exists(session, image_name):
# # # # #         print(f"Image '{image_name}' already exists in the database.")
# # # # #         return
# # # # #
# # # # #     # קריאת התמונה כ-BLOB
# # # # #     with open(image_path, 'rb') as file:
# # # # #         image_data = file.read()
# # # # #
# # # # #     # יצירת אובייקט Image חדש
# # # # #     new_image = Image(image_data=image_data, image_name=image_name)
# # # # #     session.add(new_image)
# # # # #     session.commit()
# # # # #
# # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # #         existing_face_id = face_exists(session, encoding)
# # # # #
# # # # #         if existing_face_id == -1:
# # # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # # #             face_image_data = buffer.tobytes()
# # # # #             encoded_face = pickle.dumps(encoding)
# # # # #
# # # # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # # # #             session.add(new_face)
# # # # #             session.commit()
# # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # #             session.add(link)
# # # # #         else:
# # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
# # # # #             session.add(link)
# # # # #
# # # # #     session.commit()
# # # # #     print("Image and faces uploaded successfully.")
# # # # #
# # # # # # יצירת FastAPI
# # # # # app = FastAPI()
# # # # #
# # # # #
# # # # # # חיבור למסד הנתונים
# # # # # def get_db():
# # # # #     session = create_database_connection()
# # # # #     try:
# # # # #         yield session
# # # # #     finally:
# # # # #         session.close()
# # # # #
# # # # #
# # # # # @app.get("/images/")
# # # # # def get_all_images(db: Session = Depends(get_db)):
# # # # #     images = db.query(Image).all()
# # # # #     return [{"id": image.id, "name": image.image_name} for image in images]
# # # # #
# # # # #
# # # # # @app.get("/images/{image_id}")
# # # # # def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
# # # # #     image = db.query(Image).filter(Image.id == image_id).first()
# # # # #     if not image:
# # # # #         raise HTTPException(status_code=404, detail="Image not found")
# # # # #     return {"id": image.id, "name": image.image_name}
# # # # #
# # # # #
# # # # # # @app.post("/images/")
# # # # # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # # # # #     image_path = f"/path/to/save/{file.filename}"  # Update this path
# # # # # #     with open(image_path, "wb") as buffer:
# # # # # #         buffer.write(await file.read())
# # # # # #
# # # # # #     image = load_image(image_path)
# # # # # #     face_locations = detect_faces(image)
# # # # # #     face_images = extract_faces(image, face_locations)
# # # # # #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # # # # #
# # # # # #     save_image_to_db(db, image_path, known_faces, face_images)
# # # # # #     return {"detail": "Image uploaded successfully"}
# # # # # #
# # # # # # @app.post("/images/")
# # # # # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # # # # #     # קריאת התמונה כ-BLOB
# # # # # #     image_data = await file.read()
# # # # # #
# # # # # #     # טוען את התמונה (אם נדרש)
# # # # # #     image = load_image(image_data)  # אם load_image תומכת ב-BLOB ישירות
# # # # # #     face_locations = detect_faces(image)
# # # # # #     face_images = extract_faces(image, face_locations)
# # # # # #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # # # # #
# # # # # #     # שמירה במסד הנתונים
# # # # # #     save_image_to_db(db, image_data, known_faces, face_images)
# # # # # #     return {"detail": "Image uploaded successfully"}
# # # # #
# # # # # @app.post("/images/")
# # # # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # # # #     try:
# # # # #         image_data = await file.read()
# # # # #         image = load_image(image_data)
# # # # #         face_locations = detect_faces(image)
# # # # #         face_images = extract_faces(image, face_locations)
# # # # #         known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # # # #
# # # # #         save_image_to_db(db, image_data, known_faces, face_images)
# # # # #         return {"detail": "Image uploaded successfully"}
# # # # #     except Exception as e:
# # # # #         print(f"Error: {e}")  # הדפסת השגיאה לקונסול
# # # # #         raise HTTPException(status_code=500, detail="Internal Server Error")
# # # # #
# # # # #
# # # # # @app.get("/faces/")
# # # # # def get_all_faces(db: Session = Depends(get_db)):
# # # # #     existing_faces = db.query(Face).all()
# # # # #     return [{"id": face.id} for face in existing_faces]
# # # # #
# # # # #
# # # # # @app.get("/faces/{face_id}")
# # # # # def get_face_by_id(face_id: int, db: Session = Depends(get_db)):
# # # # #     face = db.query(Face).filter(Face.id == face_id).first()
# # # # #     if not face:
# # # # #         raise HTTPException(status_code=404, detail="Face not found")
# # # # #     return {"id": face.id}
# # # # #
# # # # #
# # # # # @app.get("/faces/{face_id}/images/")
# # # # # def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
# # # # #     images = get_images_by_face_id(db, face_id)
# # # # #     return [{"id": image[0], "name": image[1]} for image in images]
# # # # #
# # # # #
# # # # # if __name__ == "__main__":
# # # # #     import uvicorn
# # # # #
# # # # #     uvicorn.run(app, host="127.0.0.1", port=8000)
# # # #
# # # #
# # # #
# # # #
# # # # import pickle
# # # # import cv2
# # # # import numpy as np
# # # # import face_recognition
# # # # import matplotlib.pyplot as plt
# # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
# # # # from sqlalchemy.dialects.mysql import MEDIUMBLOB
# # # # from contextlib import contextmanager
# # # # from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
# # # # import io
# # # #
# # # # # הגדרת בסיס המודל
# # # # Base = declarative_base()
# # # #
# # # # class Image(Base):
# # # #     __tablename__ = 'images'
# # # #     id = Column(Integer, primary_key=True)
# # # #     image_data = Column(MEDIUMBLOB, nullable=False)
# # # #     image_name = Column(String(255), nullable=False)
# # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # #
# # # # class Face(Base):
# # # #     __tablename__ = 'faces'
# # # #     id = Column(Integer, primary_key=True)
# # # #     encoding = Column(MEDIUMBLOB, nullable=False)
# # # #     face_image_data = Column(MEDIUMBLOB, nullable=False)
# # # #     images = relationship("ImageFaceLink", back_populates="face")
# # # #
# # # # class ImageFaceLink(Base):
# # # #     __tablename__ = 'image_face_link'
# # # #     id = Column(Integer, primary_key=True)
# # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # #     image = relationship("Image", back_populates="face_links")
# # # #     face = relationship("Face", back_populates="images")
# # # #
# # # # # חיבור למסד הנתונים
# # # # def create_database_connection():
# # # #     try:
# # # #         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image4")
# # # #         Base.metadata.create_all(engine)
# # # #         return sessionmaker(bind=engine)()
# # # #     except Exception as e:
# # # #         print(f"Error connecting to the database: {e}")
# # # #         return None
# # # #
# # # # @contextmanager
# # # # def get_session():
# # # #     session = create_database_connection()
# # # #     try:
# # # #         yield session
# # # #     finally:
# # # #         session.close()
# # # #
# # # # # פונקציות לעיבוד תמונה
# # # # def load_image(image_data):
# # # #     nparr = np.frombuffer(image_data, np.uint8)
# # # #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# # # #     return image
# # # #
# # # # def detect_faces(image):
# # # #     return face_recognition.face_locations(image, model="hog")
# # # #
# # # # def extract_faces(image, face_locations):
# # # #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# # # #
# # # # def display_faces(face_images):
# # # #     if not face_images:
# # # #         print("No faces detected.")
# # # #         return
# # # #     plt.figure(figsize=(10, 10))
# # # #     for i, face in enumerate(face_images):
# # # #         plt.subplot(1, len(face_images), i + 1)
# # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # #         plt.axis('off')
# # # #     plt.show()
# # # #
# # # # def resize_image(image, scale_percent):
# # # #     width = int(image.shape[1] * scale_percent / 100)
# # # #     height = int(image.shape[0] * scale_percent / 100)
# # # #     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
# # # #
# # # # # פונקציות לזיהוי פנים
# # # # def recognize_faces(image, session, known_faces, threshold=0.5):
# # # #     image_resized = resize_image(image, 20)
# # # #     face_locations = detect_faces(image_resized)
# # # #     recognized_faces_indices = []
# # # #
# # # #     for face_location in face_locations:
# # # #         (top, right, bottom, left) = face_location
# # # #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# # # #
# # # #         existing_face_id = face_exists(session, face_encoding, threshold)
# # # #         recognized_faces_indices.append(existing_face_id)
# # # #
# # # #         if existing_face_id == -1:
# # # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
# # # #         else:
# # # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
# # # #             cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# # # #                         (255, 0, 0), 2)
# # # #
# # # #     return image_resized, recognized_faces_indices
# # # #
# # # # def get_images_by_face_id(session, face_id):
# # # #     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# # # #     images = []
# # # #     for link in links:
# # # #         image = session.query(Image).filter(Image.id == link.image_id).first()
# # # #         if image:
# # # #             images.append((image.id, image.image_name, image.image_data))
# # # #     return images
# # # #
# # # # def display_images(images):
# # # #     plt.figure(figsize=(10, 10))
# # # #     for i, (image_id, image_name, image_data) in enumerate(images):
# # # #         image_array = np.frombuffer(image_data, np.uint8)
# # # #         image_decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# # # #         plt.subplot(1, len(images), i + 1)
# # # #         plt.imshow(cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB))
# # # #         plt.title(image_name)
# # # #         plt.axis('off')
# # # #     plt.show()
# # # #
# # # # # פונקציות למסד נתונים
# # # # def get_all_faces(session):
# # # #     existing_faces = session.query(Face).all()
# # # #     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
# # # #
# # # # def face_exists(session, new_encoding, threshold=0.6):
# # # #     face_data = get_all_faces(session)
# # # #     closest_id = -1
# # # #     closest_distance = float('inf')
# # # #
# # # #     for face_id, face_encoding in face_data:
# # # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # # #         if distance < threshold and distance < closest_distance:
# # # #             closest_distance = distance
# # # #             closest_id = face_id
# # # #
# # # #     return closest_id
# # # #
# # # # def image_exists(session, image_name):
# # # #     return session.query(Image).filter(Image.image_name == image_name).first() is not None
# # # #
# # # # def save_image_to_db(session, image_name, image_data, face_encodings, face_images):
# # # #     if image_exists(session, image_name):
# # # #         print(f"Image '{image_name}' already exists in the database.")
# # # #         return
# # # #
# # # #     new_image = Image(image_data=image_data, image_name=image_name)
# # # #     session.add(new_image)
# # # #     session.commit()
# # # #
# # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # #         existing_face_id = face_exists(session, encoding)
# # # #
# # # #         if existing_face_id == -1:
# # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # #             face_image_data = buffer.tobytes()
# # # #             encoded_face = pickle.dumps(encoding)
# # # #
# # # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # # #             session.add(new_face)
# # # #             session.commit()
# # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # #             session.add(link)
# # # #         else:
# # # #             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
# # # #             session.add(link)
# # # #
# # # #     session.commit()
# # # #     print("Image and faces uploaded successfully.")
# # # #
# # # # # יצירת FastAPI
# # # # app = FastAPI()
# # # #
# # # # def get_db():
# # # #     session = create_database_connection()
# # # #     try:
# # # #         yield session
# # # #     finally:
# # # #         session.close()
# # # #
# # # # @app.get("/images/")
# # # # def get_all_images(db: Session = Depends(get_db)):
# # # #     images = db.query(Image).all()
# # # #     return [{"id": image.id, "name": image.image_name} for image in images]
# # # #
# # # # @app.get("/images/{image_id}")
# # # # def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
# # # #     image = db.query(Image).filter(Image.id == image_id).first()
# # # #     if not image:
# # # #         raise HTTPException(status_code=404, detail="Image not found")
# # # #     return {"id": image.id, "name": image.image_name}
# # # #
# # # # @app.post("/images/")
# # # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # # #     try:
# # # #         image_data = await file.read()
# # # #         image = load_image(image_data)
# # # #         face_locations = detect_faces(image)
# # # #         face_images = extract_faces(image, face_locations)
# # # #         known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # # #
# # # #         # קח את השם מהקובץ
# # # #         image_name = file.filename
# # # #
# # # #         save_image_to_db(db, image_name, image_data, known_faces, face_images)
# # # #         return {"detail": "Image uploaded successfully"}
# # # #     except Exception as e:
# # # #         print(f"Error: {e}")  # הדפסת השגיאה לקונסול
# # # #         raise HTTPException(status_code=500, detail="Internal Server Error")
# # # #
# # # # @app.get("/faces/")
# # # # def get_all_faces(db: Session = Depends(get_db)):
# # # #     existing_faces = db.query(Face).all()
# # # #     return [{"id": face.id} for face in existing_faces]
# # # #
# # # # @app.get("/faces/{face_id}")
# # # # def get_face_by_id(face_id: int, db: Session = Depends(get_db)):
# # # #     face = db.query(Face).filter(Face.id == face_id).first()
# # # #     if not face:
# # # #         raise HTTPException(status_code=404, detail="Face not found")
# # # #     return {"id": face.id}
# # # #
# # # # @app.get("/faces/{face_id}/images/")
# # # # def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
# # # #     images = get_images_by_face_id(db, face_id)
# # # #     return [{"id": image[0], "name": image[1]} for image in images]
# # # #
# # # # if __name__ == "__main__":
# # # #     import uvicorn
# # # #     uvicorn.run(app, host="127.0.0.1", port=8000)
# # # import pickle
# # # import cv2
# # # import numpy as np
# # # import face_recognition
# # # import matplotlib.pyplot as plt
# # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
# # # from sqlalchemy.dialects.mysql import MEDIUMBLOB
# # # from contextlib import contextmanager
# # # from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
# # # import io
# # #
# # # # הגדרת בסיס המודל
# # # Base = declarative_base()
# # #
# # # class Image(Base):
# # #     __tablename__ = 'images'
# # #     id = Column(Integer, primary_key=True)
# # #     image_data = Column(MEDIUMBLOB, nullable=False)
# # #     image_name = Column(String(255), nullable=False)
# # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # #
# # # class Face(Base):
# # #     __tablename__ = 'faces'
# # #     id = Column(Integer, primary_key=True)
# # #     encoding = Column(MEDIUMBLOB, nullable=False)
# # #     face_image_data = Column(MEDIUMBLOB, nullable=False)
# # #     images = relationship("ImageFaceLink", back_populates="face")
# # #
# # # class ImageFaceLink(Base):
# # #     __tablename__ = 'image_face_link'
# # #     id = Column(Integer, primary_key=True)
# # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # #     image = relationship("Image", back_populates="face_links")
# # #     face = relationship("Face", back_populates="images")
# # #
# # # # חיבור למסד הנתונים
# # # def create_database_connection():
# # #     try:
# # #         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image4")
# # #         Base.metadata.create_all(engine)
# # #         return sessionmaker(bind=engine)()
# # #     except Exception as e:
# # #         print(f"Error connecting to the database: {e}")
# # #         return None
# # #
# # # @contextmanager
# # # def get_session():
# # #     session = create_database_connection()
# # #     try:
# # #         yield session
# # #     finally:
# # #         session.close()
# # #
# # # # פונקציות לעיבוד תמונה
# # # def load_image(image_data):
# # #     nparr = np.frombuffer(image_data, np.uint8)
# # #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# # #     return image
# # #
# # # def detect_faces(image):
# # #     return face_recognition.face_locations(image, model="hog")
# # #
# # # def extract_faces(image, face_locations):
# # #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# # #
# # # def display_faces(face_images):
# # #     if not face_images:
# # #         print("No faces detected.")
# # #         return
# # #     plt.figure(figsize=(10, 10))
# # #     for i, face in enumerate(face_images):
# # #         plt.subplot(1, len(face_images), i + 1)
# # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # #         plt.axis('off')
# # #     plt.show()
# # #
# # # def resize_image(image, scale_percent):
# # #     width = int(image.shape[1] * scale_percent / 100)
# # #     height = int(image.shape[0] * scale_percent / 100)
# # #     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
# # #
# # # # פונקציות לזיהוי פנים
# # # def recognize_faces(image, session, known_faces, threshold=0.5):
# # #     image_resized = resize_image(image, 20)
# # #     face_locations = detect_faces(image_resized)
# # #     recognized_faces_indices = []
# # #
# # #     for face_location in face_locations:
# # #         (top, right, bottom, left) = face_location
# # #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# # #
# # #         existing_face_id = face_exists(session, face_encoding, threshold)
# # #         recognized_faces_indices.append(existing_face_id)
# # #
# # #         if existing_face_id == -1:
# # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
# # #         else:
# # #             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
# # #             cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# # #                         (255, 0, 0), 2)
# # #
# # #     return image_resized, recognized_faces_indices
# # #
# # # def get_images_by_face_id(session, face_id):
# # #     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# # #     images = []
# # #     for link in links:
# # #         image = session.query(Image).filter(Image.id == link.image_id).first()
# # #         if image:
# # #             images.append((image.id, image.image_name, image.image_data))
# # #     return images
# # #
# # # def display_images(images):
# # #     plt.figure(figsize=(10, 10))
# # #     for i, (image_id, image_name, image_data) in enumerate(images):
# # #         image_array = np.frombuffer(image_data, np.uint8)
# # #         image_decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# # #         plt.subplot(1, len(images), i + 1)
# # #         plt.imshow(cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB))
# # #         plt.title(image_name)
# # #         plt.axis('off')
# # #     plt.show()
# # #
# # # # פונקציות למסד נתונים
# # # def get_all_faces(session):
# # #     existing_faces = session.query(Face).all()
# # #     print("exist")
# # #     faces = []
# # #     for face in existing_faces:
# # #         try:
# # #             encoding = pickle.loads(face.encoding)
# # #             faces.append((face.id, encoding))
# # #         except Exception as e:
# # #             print(f"Error loading face encoding for ID {face.id}: {e}")
# # #     print("faces")
# # #     print(faces)
# # #     return faces
# # #
# # # def face_exists(session, new_encoding, threshold=0.6):
# # #     face_data = get_all_faces(session)
# # #     closest_id = -1
# # #     closest_distance = float('inf')
# # #     print("for")
# # #     print(face_data)
# # #     if not face_data:  # אם אין פנים, החזר -1
# # #         return closest_id
# # #
# # #     for face_id, face_encoding in face_data:
# # #         print("dist")
# # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # #         print("other")
# # #         if distance < threshold and distance < closest_distance:
# # #             closest_distance = distance
# # #             closest_id = face_id
# # #     print("return")
# # #     return closest_id
# # #
# # # def image_exists(session, image_name):
# # #     return session.query(Image).filter(Image.image_name == image_name).first() is not None
# # #
# # # def save_image_to_db(session, image_name, image_data, face_encodings, face_images):
# # #     if image_exists(session, image_name):
# # #         print(f"Image '{image_name}' already exists in the database.")
# # #         return
# # #
# # #     new_image = Image(image_data=image_data, image_name=image_name)
# # #     session.add(new_image)
# # #     session.commit()
# # #
# # #     for encoding, face_image in zip(face_encodings, face_images):
# # #         existing_face_id = face_exists(session, encoding)
# # #
# # #         if existing_face_id == -1:
# # #             _, buffer = cv2.imencode('.jpg', face_image)
# # #             face_image_data = buffer.tobytes()
# # #             encoded_face = pickle.dumps(encoding)
# # #
# # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # #             session.add(new_face)
# # #             session.commit()
# # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # #             session.add(link)
# # #         else:
# # #             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
# # #             session.add(link)
# # #
# # #     session.commit()
# # #     print("Image and faces uploaded successfully.")
# # #
# # # # יצירת FastAPI
# # # app = FastAPI()
# # #
# # # def get_db():
# # #     session = create_database_connection()
# # #     try:
# # #         yield session
# # #     finally:
# # #         session.close()
# # # @app.get("/images/")
# # # def get_all_images(db: Session = Depends(get_db)):
# # #     images = db.query(Image).all()
# # #     return [{
# # #         "id": image.id,
# # #         "name": image.image_name,
# # #         "image_data": image.image_data  # החזרת נתוני התמונה
# # #     } for image in images]
# # # @app.get("/images/{image_id}")
# # # def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
# # #     image = db.query(Image).filter(Image.id == image_id).first()
# # #     if not image:
# # #         raise HTTPException(status_code=404, detail="Image not found")
# # #     return {
# # #         "id": image.id,
# # #         "name": image.image_name,
# # #         "image_data": image.image_data  # החזרת נתוני התמונה
# # #     }
# # #
# # # #
# # # # @app.post("/images/")
# # # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # # #     try:
# # # #         image_data = await file.read()
# # # #         image = load_image(image_data)
# # # #         face_locations = detect_faces(image)
# # # #         face_images = extract_faces(image, face_locations)
# # # #         known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # # #
# # # #         # קח את השם מהקובץ
# # # #         image_name = file.filename
# # # #
# # # #         save_image_to_db(db, image_name, image_data, known_faces, face_images)
# # # #         return {"detail": "Image uploaded successfully"}
# # # #     except Exception as e:
# # # #         print(f"Error: {e}")  # הדפסת השגיאה לקונסול
# # # #         raise HTTPException(status_code=500, detail="Internal Server Error")
# # #
# # # @app.post("/images/")
# # # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# # #     if not file:
# # #         raise HTTPException(status_code=400, detail="No file provided")
# # #
# # #     try:
# # #         image_data = await file.read()
# # #         if not image_data:
# # #             raise HTTPException(status_code=400, detail="File is empty")
# # #
# # #         image = load_image(image_data)
# # #         if image is None:
# # #             raise HTTPException(status_code=400, detail="Invalid image format")
# # #         print("load")
# # #         face_locations = detect_faces(image)
# # #         print("facelocation")
# # #         face_images = extract_faces(image, face_locations)
# # #         print('face_images')
# # #         if not face_images:
# # #             raise HTTPException(status_code=400, detail="No faces detected in the image")
# # #
# # #         known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # #
# # #         # קח את השם מהקובץ
# # #         image_name = file.filename
# # #
# # #         save_image_to_db(db, image_name, image_data, known_faces, face_images)
# # #         return {"detail": "Image uploaded successfully"}
# # #
# # #     except HTTPException as http_exception:
# # #         raise http_exception  # אם זו כבר HTTPException, פשוט להעלות אותה מחדש
# # #     except Exception as e:
# # #         print(f"Error: {e}")  # הדפסת השגיאה לקונסול
# # #
# # #         raise HTTPException(status_code=500, detail="Internal Server Error")
# # #
# # #
# # # @app.get("/faces/")
# # # def get_all_faces(db: Session = Depends(get_db)):
# # #     existing_faces = db.query(Face).all()
# # #     faces_data = []
# # #
# # #     for face in existing_faces:
# # #         face_image_data = face.face_image_data  # הנח שהשדה הזה מכיל את נתוני התמונה
# # #         faces_data.append({
# # #             "id": face.id,
# # #             "image_data": face_image_data  # החזרת נתוני התמונה
# # #         })
# # #
# # #     return faces_data
# # #
# # #
# # # @app.get("/faces/{face_id}")
# # # def get_face_by_id(face_id: int, db: Session = Depends(get_db)):
# # #     face = db.query(Face).filter(Face.id == face_id).first()
# # #     if not face:
# # #         raise HTTPException(status_code=404, detail="Face not found")
# # #
# # #     # החזרת מידע על הפנים כולל נתוני התמונה
# # #     return {
# # #         "id": face.id,
# # #         "face_image_data": face.face_image_data,
# # #         "encoding": face.encoding,
# # #         "linked_images": [
# # #             {
# # #                 "id": link.image.id,
# # #                 "image_name": link.image.image_name,
# # #                 "image_data": link.image.image_data
# # #             }
# # #             for link in face.images
# # #         ]
# # #     }
# # #
# # #
# # # @app.get("/faces/{face_id}/images/")
# # # def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
# # #     # קבלת כל הקישורים בין תמונות לפנים
# # #     links = db.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# # #
# # #     # החזרת רשימת התמונות הקשורות לפנים
# # #     return [{
# # #         "id": link.image.id,
# # #         "image_name": link.image.image_name,
# # #         "image_data": link.image.image_data
# # #     } for link in links]
# # #
# # #
# # # if __name__ == "__main__":
# # #     import uvicorn
# # #     uvicorn.run(app, host="127.0.0.1", port=8000)
# #
# #
# #
# #
# # import numpy as np
# # import pickle
# # import cv2
# # from fastapi import FastAPI, Depends, HTTPException, File, UploadFile
# # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # from sqlalchemy.orm import sessionmaker, relationship, Session
# # from sqlalchemy.ext.declarative import declarative_base
# # from contextlib import contextmanager
# # from sqlalchemy.dialects.mysql import MEDIUMBLOB
# # import face_recognition
# # import matplotlib.pyplot as plt
# #
# # # הגדרת בסיס המודל
# # Base = declarative_base()
# #
# # class Image(Base):
# #     __tablename__ = 'images'
# #     id = Column(Integer, primary_key=True)
# #     image_data = Column(MEDIUMBLOB, nullable=False)
# #     image_name = Column(String(255), nullable=False)
# #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# #
# # class Face(Base):
# #     __tablename__ = 'faces'
# #     id = Column(Integer, primary_key=True)
# #     encoding = Column(MEDIUMBLOB, nullable=False)
# #     face_image_data = Column(MEDIUMBLOB, nullable=False)
# #     images = relationship("ImageFaceLink", back_populates="face")
# #
# # class ImageFaceLink(Base):
# #     __tablename__ = 'image_face_link'
# #     id = Column(Integer, primary_key=True)
# #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# #     image = relationship("Image", back_populates="face_links")
# #     face = relationship("Face", back_populates="images")
# #
# # # חיבור למסד הנתונים
# # def create_database_connection():
# #     try:
# #         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image4")
# #         Base.metadata.create_all(engine)
# #         return sessionmaker(bind=engine)()
# #     except Exception as e:
# #         print(f"Error connecting to the database: {e}")
# #         return None
# #
# # @contextmanager
# # def get_session():
# #     session = create_database_connection()
# #     try:
# #         yield session
# #     finally:
# #         session.close()
# #
# # # פונקציות לעיבוד תמונה
# # def load_image(image_data):
# #     nparr = np.frombuffer(image_data, np.uint8)
# #     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# #     return image
# #
# # def detect_faces(image):
# #     return face_recognition.face_locations(image, model="hog")
# #
# # def extract_faces(image, face_locations):
# #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# #
# # def display_faces(face_images):
# #     if not face_images:
# #         print("No faces detected.")
# #         return
# #     plt.figure(figsize=(10, 10))
# #     for i, face in enumerate(face_images):
# #         plt.subplot(1, len(face_images), i + 1)
# #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# #         plt.axis('off')
# #     plt.show()
# #
# # def resize_image(image, scale_percent):
# #     width = int(image.shape[1] * scale_percent / 100)
# #     height = int(image.shape[0] * scale_percent / 100)
# #     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
# #
# # # פונקציות לזיהוי פנים
# # def recognize_faces(image, session, threshold=0.5):
# #     image_resized = resize_image(image, 20)
# #     face_locations = detect_faces(image_resized)
# #     recognized_faces_indices = []
# #
# #     for face_location in face_locations:
# #         (top, right, bottom, left) = face_location
# #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])
# #
# #         if face_encoding:  # בדוק אם קידוד נמצא
# #             face_encoding = face_encoding[0]
# #             existing_face_id = face_exists(session, face_encoding, threshold)
# #             recognized_faces_indices.append(existing_face_id)
# #
# #             if existing_face_id == -1:
# #                 cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
# #             else:
# #                 cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
# #                 cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# #                             (255, 0, 0), 2)
# #
# #     return image_resized, recognized_faces_indices
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
# # def display_images(images):
# #     plt.figure(figsize=(10, 10))
# #     for i, (image_id, image_name, image_data) in enumerate(images):
# #         image_array = np.frombuffer(image_data, np.uint8)
# #         image_decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# #         plt.subplot(1, len(images), i + 1)
# #         plt.imshow(cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB))
# #         plt.title(image_name)
# #         plt.axis('off')
# #     plt.show()
# #
# # # פונקציות למסד נתונים
# # def get_all_faces(session):
# #     existing_faces = session.query(Face).all()
# #     faces = []
# #     for face in existing_faces:
# #         try:
# #             encoding = pickle.loads(face.encoding)
# #             faces.append((face.id, encoding))
# #         except Exception as e:
# #             print(f"Error loading face encoding for ID {face.id}: {e}")
# #     return faces
# #
# # def face_exists(session, new_encoding, threshold=0.6):
# #     face_data = get_all_faces(session)
# #     closest_id = -1
# #     closest_distance = float('inf')
# #
# #     if not face_data:  # אם אין פנים, החזר -1
# #         return closest_id
# #
# #     for face_id, face_encoding in face_data:
# #         existing_face_encoding = np.array(pickle.loads(face_encoding))  # המרת קידוד מ-pickle למערך NumPy
# #         new_encoding = np.array(face_encoding)  # ודא שזה גם מערך NumPy
# #
# #         distance = np.linalg.norm(existing_face_encoding - new_encoding)
# #
# #         # distance = face_recognition.face_distance(pickle.loads(face_encoding), new_encoding)[0]
# #         #distance = face_recognition.face_distance(np.array([face_encoding]), np.array(new_encoding))[0]
# #         if distance < threshold and distance < closest_distance:
# #             closest_distance = distance
# #             closest_id = face_id
# #     return closest_id
# #
# # def image_exists(session, image_name):
# #     return session.query(Image).filter(Image.image_name == image_name).first() is not None
# #
# # def save_image_to_db(session, image_name, image_data, face_encodings, face_images):
# #     if image_exists(session, image_name):
# #         print(f"Image '{image_name}' already exists in the database.")
# #         return
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
# #     print("Image and faces uploaded successfully.")
# #
# # # יצירת FastAPI
# # app = FastAPI()
# #
# # def get_db():
# #     session = create_database_connection()
# #     try:
# #         yield session
# #     finally:
# #         session.close()
# #
# # @app.get("/images/")
# # def get_all_images(db: Session = Depends(get_db)):
# #     images = db.query(Image).all()
# #     return [{
# #         "id": image.id,
# #         "name": image.image_name,
# #         "image_data": image.image_data  # החזרת נתוני התמונה
# #     } for image in images]
# #
# # @app.get("/images/{image_id}")
# # def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
# #     image = db.query(Image).filter(Image.id == image_id).first()
# #     if not image:
# #         raise HTTPException(status_code=404, detail="Image not found")
# #     return {
# #         "id": image.id,
# #         "name": image.image_name,
# #         "image_data": image.image_data  # החזרת נתוני התמונה
# #     }
# #
# # @app.post("/images/")
# # async def add_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
# #     if not file:
# #         raise HTTPException(status_code=400, detail="No file provided")
# #
# #     try:
# #         image_data = await file.read()
# #         if not image_data:
# #             raise HTTPException(status_code=400, detail="File is empty")
# #
# #         image = load_image(image_data)
# #         if image is None:
# #             raise HTTPException(status_code=400, detail="Invalid image format")
# #         face_locations = detect_faces(image)
# #         if not face_locations:
# #             raise HTTPException(status_code=400, detail="No faces detected in the image")
# #
# #         known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# #
# #         # קח את השם מהקובץ
# #         image_name = file.filename
# #
# #         save_image_to_db(db, image_name, image_data, known_faces, extract_faces(image, face_locations))
# #         return {"detail": "Image uploaded successfully"}
# #
# #     except HTTPException as http_exception:
# #         raise http_exception  # אם זו כבר HTTPException, פשוט להעלות אותה מחדש
# #     except Exception as e:
# #         print(f"Error: {e}")  # הדפסת השגיאה לקונסול
# #         raise HTTPException(status_code=500, detail="Internal Server Error")
# #
# # @app.get("/faces/")
# # def get_all_faces(db: Session = Depends(get_db)):
# #     existing_faces = db.query(Face).all()
# #     faces_data = []
# #
# #     for face in existing_faces:
# #         face_image_data = face.face_image_data  # הנח שהשדה הזה מכיל את נתוני התמונה
# #         faces_data.append({
# #             "id": face.id,
# #             "image_data": face_image_data  # החזרת נתוני התמונה
# #         })
# #
# #     return faces_data
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
# #
# # @app.get("/faces/{face_id}/images/")
# # def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
# #     # קבלת כל הקישורים בין תמונות לפנים
# #     links = db.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
# #
# #     # החזרת רשימת התמונות הקשורות לפנים
# #     return [{
# #         "id": link.image.id,
# #         "image_name": link.image.image_name,
# #         "image_data": link.image.image_data
# #     } for link in links]
# #
# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="127.0.0.1", port=8000)
#
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
# import base64
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
#         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image4")
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
# def get_images_by_face_id(session, face_id):
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
# # # הגדרת הגדרות CORS
# # origins = [
# #     "http://localhost:5173",  # דוגמה לדומיין שאתה רוצה לאפשר
# #     "http://localhost:5174",  # דוגמה לדומיין שאתה רוצה לאפשר
# #     "http://localhost:5175",  # דוגמה לדומיין שאתה רוצה לאפשר
# #     "https://example.com",     # דומיין נוסף
# # ]
# #
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,  # רשימת הדומיינים המורשים
# #     allow_credentials=True,
# #     allow_methods=["*"],  # כל השיטות (GET, POST, וכו')
# #     allow_headers=["*"],   # כל הכותרות
# # )
#
#
# def get_db():
#     session = create_database_connection()
#     try:
#         yield session
#     finally:
#         session.close()
#
# # @app.get("/images/")
# # def get_all_images(db: Session = Depends(get_db)):
# #     images = db.query(Image).all()
# #     return [{
# #         "id": image.id,
# #         "name": image.image_name,
# #         "image_data": image.image_data  # החזרת נתוני התמונה
# #     } for image in images]
# @app.get("/images/")
# def get_all_images(db: Session= Depends(get_db)):
#     images = db.query(Image).all()
#     return [{
#         "id": image.id,
#         "name": image.image_name,
#         "image_data": image.image_data
#     } for image in images]
# @app.get("/images/{image_id}")
# def get_image_by_id(image_id: int, db: Session = Depends(get_db)):
#     image = db.query(Image).filter(Image.id == image_id).first()
#     if not image:
#         raise HTTPException(status_code=404, detail="Image not found")
#     return {
#         "id": image.id,
#         "name": image.image_name,
#         "image_data": image.image_data  # החזרת נתוני התמונה
#     }
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
#         face_image_data = face.face_image_data  # הנח שהשדה הזה מכיל את נתוני התמונה
#         faces_data.append({
#             "id": face.id,
#             "image_data": face_image_data  # החזרת נתוני התמונה
#         })
#
#     return faces_data
#
# @app.get("/faces/{face_id}")
# def get_face_by_id(face_id: int, db: Session = Depends(get_db)):
#     face = db.query(Face).filter(Face.id == face_id).first()
#     if not face:
#         raise HTTPException(status_code=404, detail="Face not found")
#
#     # החזרת מידע על הפנים כולל נתוני התמונה
#     return {
#         "id": face.id,
#         "face_image_data": face.face_image_data,
#         "encoding": face.encoding,
#         "linked_images": [
#             {
#                 "id": link.image.id,
#                 "image_name": link.image.image_name,
#                 "image_data": link.image.image_data
#             }
#             for link in face.images
#         ]
#     }
#
# @app.get("/faces/{face_id}/images/")
# def get_images_by_face_id(face_id: int, db: Session = Depends(get_db)):
#     # קבלת כל הקישורים בין תמונות לפנים
#     links = db.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
#
#     # החזרת רשימת התמונות הקשורות לפנים
#     return [{
#         "id": link.image.id,
#         "image_name": link.image.image_name,
#         "image_data": link.image.image_data
#     } for link in links]
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)
