# import pickle
# import cv2
# import numpy as np
# import face_recognition
# import matplotlib.pyplot as plt
# from sqlalchemy.dialects.mysql import LONGBLOB
# from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# from contextlib import contextmanager
#
#
# # הגדרת בסיס המודל
# Base = declarative_base()
#
# class Image(Base):
#     __tablename__ = 'images'
#     id = Column(Integer, primary_key=True)
#     image_data = Column(LONGBLOB, nullable=False)
#     image_name = Column(String(255), nullable=False)
#     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
#
# class Face(Base):
#     __tablename__ = 'faces'
#     id = Column(Integer, primary_key=True)
#     encoding = Column(LONGBLOB, nullable=False)
#     face_image_data = Column(LONGBLOB, nullable=False)
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
#         engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
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
# def load_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
#     return image
#
# def detect_faces(image):
#     return face_recognition.face_locations(image, model="hog")
#
# def extract_faces(image, face_locations):
#     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
#
# def display_faces(face_images):
#     if not face_images:
#         print("No faces detected.")
#         return
#     plt.figure(figsize=(10, 10))
#     for i, face in enumerate(face_images):
#         plt.subplot(1, len(face_images), i + 1)
#         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#     plt.show()
#
# def resize_image(image, scale_percent):
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
#
# # פונקציות לזיהוי פנים
# def recognize_faces(image, session, known_faces, threshold=0.5):
#     image_resized = resize_image(image, 20)
#     face_locations = detect_faces(image_resized)
#     recognized_faces_indices = []
#
#     for face_location in face_locations:
#         (top, right, bottom, left) = face_location
#         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
#
#         # שימוש בפונקציה face_exists כדי לבדוק אם הפנים כבר קיימות
#         existing_face_id = face_exists(session, face_encoding, threshold)
#         recognized_faces_indices.append(existing_face_id)
#
#         # אם הפנים לא קיימות, ניתן להוסיף קוד לצייר מלבן סביב הפנים
#         if existing_face_id == -1:
#             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
#         else:
#             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
#             cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                         (255, 0, 0), 2)
#
#     return image_resized, recognized_faces_indices
# ###
# def get_images_by_face_id(session, face_id):
#     """Retrieve all images associated with a specific face_id."""
#     links = session.query(ImageFaceLink).filter(ImageFaceLink.face_id == face_id).all()
#     images = []
#     for link in links:
#         image = session.query(Image).filter(Image.id == link.image_id).first()
#         if image:
#             images.append((image.id, image.image_name, image.image_data))
#     return images
# def display_images(images):
#     """Display images given a list of image tuples (id, name, data)."""
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
#
# # פונקציות למסד נתונים
# def get_all_faces(session):
#     existing_faces = session.query(Face).all()
#     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
#
# def face_exists(session, new_encoding, threshold=0.6):
#     face_data = get_all_faces(session)
#     closest_id = -1
#     closest_distance = float('inf')
#
#     for face_id, face_encoding in face_data:
#         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
#         if distance < threshold and distance < closest_distance:
#             closest_distance = distance
#             closest_id = face_id
#
#     return closest_id
#
# def image_exists(session, image_name):
#     """Check if an image with the given name already exists in the database."""
#     return session.query(Image).filter(Image.image_name == image_name).first() is not None
#
#
# def save_image_to_db(session, image_path, face_encodings, face_images):
#     image_name = image_path.split('/')[-1]
#
#     # בדוק אם התמונה כבר קיימת
#     if image_exists(session, image_name):
#         print(f"Image '{image_name}' already exists in the database.")
#         return
#
#     with open(image_path, 'rb') as file:
#         image_data = file.read()
#
#     new_image = Image(image_data=image_data, image_name=image_name)
#     session.add(new_image)
#     session.commit()
#
#     for encoding, face_image in zip(face_encodings, face_images):
#         existing_face_id = face_exists(session, encoding)
#
#         if existing_face_id == -1:  # אם הפנים לא קיימות
#             _, buffer = cv2.imencode('.jpg', face_image)
#             face_image_data = buffer.tobytes()
#             encoded_face = pickle.dumps(encoding)
#
#             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
#             session.add(new_face)
#             session.commit()
#             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
#             session.add(link)
#         else:  # אם הפנים כבר קיימות
#             link = ImageFaceLink(image_id=new_image.id, face_id=existing_face_id)
#             session.add(link)
#
#     session.commit()
#     print("Image and faces uploaded successfully.")
#
#
# def main():
#     with get_session() as session:
#         original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
#         image = load_image(original_image_path)
#         face_locations = detect_faces(image)
#         face_images = extract_faces(image, face_locations)
#
#         display_faces(face_images)
#
#         if face_images:
#             known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in
#                            face_locations]
#             save_image_to_db(session, original_image_path, known_faces, face_images)
#
#             new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
#             new_image = load_image(new_image_path)
#             recognized_image, recognized_faces_indices = recognize_faces(new_image, session, known_faces)
#
#             cv2.imshow("Recognized Faces", recognized_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#             print("Recognized Indices: ", recognized_faces_indices)
#
#             # קבלת התמונות לפי face_id
#             for face_id in recognized_faces_indices:
#                 if face_id != -1:  # אם הפנים זוהו
#                     images = get_images_by_face_id(session, face_id)
#                     print(f"Images for face ID {face_id}:")
#                     display_images(images)
#
#
#
#
#
# if __name__ == "__main__":
#     main()
