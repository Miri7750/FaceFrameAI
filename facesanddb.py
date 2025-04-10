# # # # # # # import cv2
# # # # # # # import numpy as np
# # # # # # # import face_recognition
# # # # # # # import matplotlib.pyplot as plt
# # # # # # # from sqlalchemy.dialects.mysql import LONGBLOB
# # # # # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # # # # # #
# # # # # # # # הגדרת בסיס המודל
# # # # # # # Base = declarative_base()
# # # # # # #
# # # # # # # # הגדרת המודל
# # # # # # # # class Image(Base):
# # # # # # # #     __tablename__ = 'images'
# # # # # # # #
# # # # # # # #     id = Column(Integer, primary_key=True)
# # # # # # # #     image_data = Column(LONGBLOB, nullable=False)  # שינוי ל-LONGBLOB
# # # # # # # #     image_name = Column(String(255), nullable=False)
# # # # # # # #     faces = relationship("Face", back_populates="image", cascade="all, delete-orphan")
# # # # # # # #
# # # # # # # # class Face(Base):
# # # # # # # #     __tablename__ = 'faces'
# # # # # # # #
# # # # # # # #     id = Column(Integer, primary_key=True)
# # # # # # # #     encoding = Column(LONGBLOB, nullable=False)  # קידוד הפנים
# # # # # # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)  # קשר לתמונה
# # # # # # # #     image = relationship("Image", back_populates="faces")  # קשר הפוך לתמונה
# # # # # # #
# # # # # # # class Image(Base):
# # # # # # #     __tablename__ = 'images'
# # # # # # #
# # # # # # #     id = Column(Integer, primary_key=True)
# # # # # # #     image_data = Column(LONGBLOB, nullable=False)
# # # # # # #     image_name = Column(String(255), nullable=False)
# # # # # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # # # # #
# # # # # # #
# # # # # # # class Face(Base):
# # # # # # #     __tablename__ = 'faces'
# # # # # # #
# # # # # # #     id = Column(Integer, primary_key=True)
# # # # # # #     encoding = Column(LONGBLOB, nullable=False)  # קידוד הפנים
# # # # # # #     face_image_data = Column(LONGBLOB, nullable=False)  # תמונה חתוכה של הפנים
# # # # # # #     images = relationship("ImageFaceLink", back_populates="face")  # קשר עם טבלת הקישור
# # # # # # #
# # # # # # # class ImageFaceLink(Base):
# # # # # # #     __tablename__ = 'image_face_link'
# # # # # # #
# # # # # # #     id = Column(Integer, primary_key=True)
# # # # # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # # # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # # # # #     image = relationship("Image", back_populates="face_links")
# # # # # # #     face = relationship("Face", back_populates="image_links")
# # # # # # #
# # # # # # # # עדכון המודל Image
# # # # # # # Image.face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # # # # #
# # # # # # # # חיבור למסד הנתונים
# # # # # # # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image")
# # # # # # # Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
# # # # # # #
# # # # # # # # יצירת סשן
# # # # # # # Session = sessionmaker(bind=engine)
# # # # # # # session = Session()
# # # # # # #
# # # # # # # def load_image(image_path):
# # # # # # #     """טוען תמונה מכתובת נתונה."""
# # # # # # #     image = cv2.imread(image_path)
# # # # # # #     if image is None:
# # # # # # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # # # # # #     return image
# # # # # # #
# # # # # # # def detect_faces(image):
# # # # # # #     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
# # # # # # #     return face_recognition.face_locations(image, model="hog")  # שימוש במודל HOG לזיהוי מדויק יותר
# # # # # # #
# # # # # # # def extract_faces(image, face_locations):
# # # # # # #     """חותך את הפנים מהתמונה ומחזיר רשימה של תמונות הפנים."""
# # # # # # #     face_images = []
# # # # # # #     for (top, right, bottom, left) in face_locations:
# # # # # # #         face_image = image[top:bottom, left:right]
# # # # # # #         face_images.append(face_image)
# # # # # # #     return face_images
# # # # # # #
# # # # # # # def display_faces(face_images):
# # # # # # #     """מציג את הפנים שזוהו."""
# # # # # # #     plt.figure(figsize=(10, 10))
# # # # # # #     for i, face in enumerate(face_images):
# # # # # # #         plt.subplot(1, len(face_images), i + 1)
# # # # # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # # # # #         plt.axis('off')
# # # # # # #     plt.show()
# # # # # # #
# # # # # # # def resize_image(image, scale_percent):
# # # # # # #     """משנה את גודל התמונה לפי אחוז נתון."""
# # # # # # #     width = int(image.shape[1] * scale_percent / 100)
# # # # # # #     height = int(image.shape[0] * scale_percent / 100)
# # # # # # #     dim = (width, height)
# # # # # # #     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# # # # # # #
# # # # # # # def recognize_faces(image_b, known_faces, threshold=0.5):  # סף נמוך יותר
# # # # # # #     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה רשימה של אינדקסים של הפנים המזוהות."""
# # # # # # #     image = resize_image(image_b, 20)
# # # # # # #     face_locations = detect_faces(image)
# # # # # # #     recognized_faces_indices = []
# # # # # # #
# # # # # # #     for face_location in face_locations:
# # # # # # #         (top, right, bottom, left) = face_location
# # # # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # # # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # # # # # #
# # # # # # #         # בודק אם המרחקים קטנים מהסף
# # # # # # #         match_index = np.argmin(distances)
# # # # # # #         if distances[match_index] < threshold:
# # # # # # #             recognized_faces_indices.append(match_index)
# # # # # # #         else:
# # # # # # #             recognized_faces_indices.append(-1)
# # # # # # #
# # # # # # #         # מצייר מלבן סביב הפנים ומוסיף את האינדקס
# # # # # # #         cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
# # # # # # #         cv2.putText(image, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # # # # # #
# # # # # # #     return image, recognized_faces_indices
# # # # # # #
# # # # # # # #
# # # # # # # # def save_image_to_db(image_path, face_encodings):
# # # # # # # #     # קריאת התמונה מקובץ
# # # # # # # #     with open(image_path, 'rb') as file:
# # # # # # # #         image_data = file.read()
# # # # # # # #
# # # # # # # #     # יצירת מופע של המודל
# # # # # # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # # # # # #
# # # # # # # #     # הוספת התמונה למסד הנתונים
# # # # # # # #     session.add(new_image)
# # # # # # # #     session.commit()  # שמירת התמונה כדי לקבל את ה-ID שלה
# # # # # # # #
# # # # # # # #     # שמירת הפנים המזוהות
# # # # # # # #     for encoding in face_encodings:
# # # # # # # #         new_face = Face(encoding=encoding, image_id=new_image.id)
# # # # # # # #         session.add(new_face)
# # # # # # # #
# # # # # # # #     session.commit()
# # # # # # # #     print("Image and faces uploaded successfully.")
# # # # # # #
# # # # # # #
# # # # # # # def save_image_to_db(image_path, face_encodings, face_images):
# # # # # # #     # קריאת התמונה מקובץ
# # # # # # #     with open(image_path, 'rb') as file:
# # # # # # #         image_data = file.read()
# # # # # # #
# # # # # # #     # יצירת מופע של המודל
# # # # # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # # # # #
# # # # # # #     # הוספת התמונה למסד הנתונים
# # # # # # #     session.add(new_image)
# # # # # # #     session.commit()  # שמירת התמונה כדי לקבל את ה-ID שלה
# # # # # # #
# # # # # # #     # שמירת הפנים המזוהות
# # # # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # # # #         # שמירת התמונה החתוכה של הפנים
# # # # # # #         _, buffer = cv2.imencode('.jpg', face_image)
# # # # # # #         face_image_data = buffer.tobytes()
# # # # # # #
# # # # # # #         new_face = Face(encoding=encoding, face_image_data=face_image_data)
# # # # # # #         session.add(new_face)
# # # # # # #         session.commit()  # שמירה כדי לקבל את ה-ID של הפנים
# # # # # # #
# # # # # # #         # יצירת קשר בין התמונה לפנים
# # # # # # #         link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # # # #         session.add(link)
# # # # # # #
# # # # # # #     session.commit()
# # # # # # #     print("Image and faces uploaded successfully.")
# # # # # # #
# # # # # # #
# # # # # # # # def main():
# # # # # # # #     # טוען את התמונה המקורית
# # # # # # # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7361small.JPG"
# # # # # # # #     image = load_image(original_image_path)
# # # # # # # #
# # # # # # # #     # מזהה את הפנים בתמונה
# # # # # # # #     face_locations = detect_faces(image)
# # # # # # # #     face_images = extract_faces(image, face_locations)
# # # # # # # #
# # # # # # # #     # מציג את הפנים
# # # # # # # #     display_faces(face_images)
# # # # # # # #
# # # # # # # #     # רשימה לשמירת הפנים המוכרות
# # # # # # # #     known_faces = []
# # # # # # # #     face_encodings = []  # רשימה לשמירת קידוד הפנים
# # # # # # # #     for face_location in face_locations:
# # # # # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # # # # #         known_faces.append(face_encoding)
# # # # # # # #         face_encodings.append(face_encoding)  # שמירת הקידוד
# # # # # # # #
# # # # # # # #     # שמירת התמונה והפנים במסד הנתונים
# # # # # # # #     save_image_to_db(original_image_path, face_encodings)
# # # # # # # #
# # # # # # # #     # טוען תמונה חדשה להשוואה
# # # # # # # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7339small.JPG"
# # # # # # # #     new_image = load_image(new_image_path)
# # # # # # # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # # # # # # #
# # # # # # # #     # מציג את התמונה עם הפנים המזוהות
# # # # # # # #     cv2.imshow("Recognized Faces", recognized_image)
# # # # # # # #     cv2.waitKey(0)
# # # # # # # #     cv2.destroyAllWindows()
# # # # # # # #
# # # # # # # #     # הצגת האינדקסים המזוהות
# # # # # # # #     print("Recognized Indices: ", recognized_faces_indices)
# # # # # # #
# # # # # # #
# # # # # # # def main():
# # # # # # #     # טוען את התמונה המקורית
# # # # # # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7361small.JPG"
# # # # # # #     image = load_image(original_image_path)
# # # # # # #
# # # # # # #     # מזהה את הפנים בתמונה
# # # # # # #     face_locations = detect_faces(image)
# # # # # # #     face_images = extract_faces(image, face_locations)
# # # # # # #
# # # # # # #     # מציג את הפנים
# # # # # # #     display_faces(face_images)
# # # # # # #
# # # # # # #     # רשימה לשמירת הפנים המזוהות
# # # # # # #     known_faces = []
# # # # # # #     face_encodings = []  # רשימה לשמירת קידוד הפנים
# # # # # # #     for face_location in face_locations:
# # # # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # # # #         known_faces.append(face_encoding)
# # # # # # #         face_encodings.append(face_encoding)  # שמירת הקידוד
# # # # # # #
# # # # # # #     # שמירת התמונה והפנים במסד הנתונים
# # # # # # #     save_image_to_db(original_image_path, face_encodings, face_images)
# # # # # # #
# # # # # # #     # טוען תמונה חדשה להשוואה
# # # # # # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7339small.JPG"
# # # # # # #     new_image = load_image(new_image_path)
# # # # # # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # # # # # #
# # # # # # #     # מציג את התמונה עם הפנים המזוהות
# # # # # # #     cv2.imshow("Recognized Faces", recognized_image)
# # # # # # #     cv2.waitKey(0)
# # # # # # #     cv2.destroyAllWindows()
# # # # # # #
# # # # # # #     # הצגת האינדקסים המזוהות
# # # # # # #     print("Recognized Indices: ", recognized_faces_indices)
# # # # # # #
# # # # # # # if __name__ == "__main__":
# # # # # # #     main()
# # # # # # #
# # # # # # # # סגירת הסשן
# # # # # # # session.close()
# # # # # # #
# # # # # # # if __name__ == "__main__":
# # # # # # #     main()
# # # # # # #
# # # # # # # # סגירת הסשן
# # # # # # # session.close()
# # # # # #
# # # # # #
# # # # # # import cv2
# # # # # # import numpy as np
# # # # # # import face_recognition
# # # # # # import matplotlib.pyplot as plt
# # # # # # from sqlalchemy.dialects.mysql import LONGBLOB
# # # # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # # # # #
# # # # # # # הגדרת בסיס המודל
# # # # # # Base = declarative_base()
# # # # # #
# # # # # # class Image(Base):
# # # # # #     __tablename__ = 'images'
# # # # # #
# # # # # #     id = Column(Integer, primary_key=True)
# # # # # #     image_data = Column(LONGBLOB, nullable=False)
# # # # # #     image_name = Column(String(255), nullable=False)
# # # # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # # # #
# # # # # #
# # # # # # class Face(Base):
# # # # # #     __tablename__ = 'faces'
# # # # # #
# # # # # #     id = Column(Integer, primary_key=True)
# # # # # #     encoding = Column(LONGBLOB, nullable=False)  # קידוד הפנים
# # # # # #     face_image_data = Column(LONGBLOB, nullable=False)  # תמונה חתוכה של הפנים
# # # # # #     images = relationship("ImageFaceLink", back_populates="face")  # קשר עם טבלת הקישור
# # # # # #
# # # # # # class ImageFaceLink(Base):
# # # # # #     __tablename__ = 'image_face_link'
# # # # # #
# # # # # #     id = Column(Integer, primary_key=True)
# # # # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # # # #     image = relationship("Image", back_populates="face_links")
# # # # # #     face = relationship("Face", back_populates="images")
# # # # # #
# # # # # # # חיבור למסד הנתונים
# # # # # # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# # # # # # Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
# # # # # #
# # # # # # # יצירת סשן
# # # # # # Session = sessionmaker(bind=engine)
# # # # # # session = Session()
# # # # # #
# # # # # # def load_image(image_path):
# # # # # #     """טוען תמונה מכתובת נתונה."""
# # # # # #     image = cv2.imread(image_path)
# # # # # #     if image is None:
# # # # # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # # # # #     return image
# # # # # #
# # # # # # def detect_faces(image):
# # # # # #     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
# # # # # #     return face_recognition.face_locations(image, model="hog")  # שימוש במודל HOG לזיהוי מדויק יותר
# # # # # #
# # # # # # def extract_faces(image, face_locations):
# # # # # #     """חותך את הפנים מהתמונה ומחזיר רשימה של תמונות הפנים."""
# # # # # #     face_images = []
# # # # # #     for (top, right, bottom, left) in face_locations:
# # # # # #         face_image = image[top:bottom, left:right]
# # # # # #         face_images.append(face_image)
# # # # # #     return face_images
# # # # # #
# # # # # # def display_faces(face_images):
# # # # # #     """מציג את הפנים שזוהו."""
# # # # # #     plt.figure(figsize=(10, 10))
# # # # # #     for i, face in enumerate(face_images):
# # # # # #         plt.subplot(1, len(face_images), i + 1)
# # # # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # # # #         plt.axis('off')
# # # # # #     plt.show()
# # # # # #
# # # # # # def resize_image(image, scale_percent):
# # # # # #     """משנה את גודל התמונה לפי אחוז נתון."""
# # # # # #     width = int(image.shape[1] * scale_percent / 100)
# # # # # #     height = int(image.shape[0] * scale_percent / 100)
# # # # # #     dim = (width, height)
# # # # # #     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# # # # # #
# # # # # # def recognize_faces(image_b, known_faces, threshold=0.5):  # סף נמוך יותר
# # # # # #     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה רשימה של אינדקסים של הפנים המזוהות."""
# # # # # #     image = resize_image(image_b, 20)
# # # # # #     face_locations = detect_faces(image)
# # # # # #     recognized_faces_indices = []
# # # # # #
# # # # # #     for face_location in face_locations:
# # # # # #         (top, right, bottom, left) = face_location
# # # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # # # # #
# # # # # #         # בודק אם המרחקים קטנים מהסף
# # # # # #         match_index = np.argmin(distances)
# # # # # #         if distances[match_index] < threshold:
# # # # # #             recognized_faces_indices.append(match_index)
# # # # # #         else:
# # # # # #             recognized_faces_indices.append(-1)
# # # # # #
# # # # # #         # מצייר מלבן סביב הפנים ומוסיף את האינדקס
# # # # # #         cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
# # # # # #         cv2.putText(image, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # # # # #
# # # # # #     return image, recognized_faces_indices
# # # # # #
# # # # # # def save_image_to_db(image_path, face_encodings, face_images):
# # # # # #     # קריאת התמונה מקובץ
# # # # # #     with open(image_path, 'rb') as file:
# # # # # #         image_data = file.read()
# # # # # #
# # # # # #     # יצירת מופע של המודל
# # # # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # # # #
# # # # # #     # הוספת התמונה למסד הנתונים
# # # # # #     session.add(new_image)
# # # # # #     session.commit()  # שמירת התמונה כדי לקבל את ה-ID שלה
# # # # # #
# # # # # #     # שמירת הפנים המזוהות
# # # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # # #         # שמירת התמונה החתוכה של הפנים
# # # # # #         _, buffer = cv2.imencode('.jpg', face_image)
# # # # # #         face_image_data = buffer.tobytes()
# # # # # #
# # # # # #         new_face = Face(encoding=encoding, face_image_data=face_image_data)
# # # # # #         session.add(new_face)
# # # # # #         session.commit()  # שמירה כדי לקבל את ה-ID של הפנים
# # # # # #
# # # # # #         # יצירת קשר בין התמונה לפנים
# # # # # #         link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # # #         session.add(link)
# # # # # #
# # # # # #     session.commit()
# # # # # #     print("Image and faces uploaded successfully.")
# # # # # #
# # # # # # def main():
# # # # # #     # טוען את התמונה המקורית
# # # # # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# # # # # #     image = load_image(original_image_path)
# # # # # #
# # # # # #     # מזהה את הפנים בתמונה
# # # # # #     face_locations = detect_faces(image)
# # # # # #     face_images = extract_faces(image, face_locations)
# # # # # #
# # # # # #     # מציג את הפנים
# # # # # #     display_faces(face_images)
# # # # # #
# # # # # #     # רשימה לשמירת הפנים המזוהות
# # # # # #     known_faces = []
# # # # # #     face_encodings = []  # רשימה לשמירת קידוד הפנים
# # # # # #     for face_location in face_locations:
# # # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # # #         known_faces.append(face_encoding)
# # # # # #         face_encodings.append(face_encoding)  # שמירת הקידוד
# # # # # #
# # # # # #     # שמירת התמונה והפנים במסד הנתונים
# # # # # #     save_image_to_db(original_image_path, face_encodings, face_images)
# # # # # #
# # # # # #     # טוען תמונה חדשה להשוואה
# # # # # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# # # # # #     new_image = load_image(new_image_path)
# # # # # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # # # # #
# # # # # #     # מציג את התמונה עם הפנים המזוהות
# # # # # #     cv2.imshow("Recognized Faces", recognized_image)
# # # # # #     cv2.waitKey(0)
# # # # # #     cv2.destroyAllWindows()
# # # # # #
# # # # # #     # הצגת האינדקסים המזוהות
# # # # # #     print("Recognized Indices: ", recognized_faces_indices)
# # # # # #
# # # # # # if __name__ == "__main__":
# # # # # #     main()
# # # # # #
# # # # # # # סגירת הסשן
# # # # # # session.close()
# # # # #
# # # # #
# # # # # import pickle
# # # # # import cv2
# # # # # import numpy as np
# # # # # import face_recognition
# # # # # import matplotlib.pyplot as plt
# # # # # from sqlalchemy.dialects.mysql import LONGBLOB
# # # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # # # #
# # # # # # הגדרת בסיס המודל
# # # # # Base = declarative_base()
# # # # #
# # # # # class Image(Base):
# # # # #     __tablename__ = 'images'
# # # # #
# # # # #     id = Column(Integer, primary_key=True)
# # # # #     image_data = Column(LONGBLOB, nullable=False)
# # # # #     image_name = Column(String(255), nullable=False)
# # # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # # #
# # # # #
# # # # # class Face(Base):
# # # # #     __tablename__ = 'faces'
# # # # #
# # # # #     id = Column(Integer, primary_key=True)
# # # # #     encoding = Column(LONGBLOB, nullable=False)  # קידוד הפנים
# # # # #     face_image_data = Column(LONGBLOB, nullable=False)  # תמונה חתוכה של הפנים
# # # # #     images = relationship("ImageFaceLink", back_populates="face")  # קשר עם טבלת הקישור
# # # # #
# # # # # class ImageFaceLink(Base):
# # # # #     __tablename__ = 'image_face_link'
# # # # #
# # # # #     id = Column(Integer, primary_key=True)
# # # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # # #     image = relationship("Image", back_populates="face_links")
# # # # #     face = relationship("Face", back_populates="images")
# # # # #
# # # # # # חיבור למסד הנתונים
# # # # # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# # # # # Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
# # # # #
# # # # # # יצירת סשן
# # # # # Session = sessionmaker(bind=engine)
# # # # # session = Session()
# # # # #
# # # # # def load_image(image_path):
# # # # #     """טוען תמונה מכתובת נתונה."""
# # # # #     image = cv2.imread(image_path)
# # # # #     if image is None:
# # # # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # # # #     return image
# # # # #
# # # # # def detect_faces(image):
# # # # #     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
# # # # #     return face_recognition.face_locations(image, model="hog")  # שימוש במודל HOG לזיהוי מדויק יותר
# # # # #
# # # # # def extract_faces(image, face_locations):
# # # # #     """חותך את הפנים מהתמונה ומחזיר רשימה של תמונות הפנים."""
# # # # #     face_images = []
# # # # #     for (top, right, bottom, left) in face_locations:
# # # # #         face_image = image[top:bottom, left:right]
# # # # #         face_images.append(face_image)
# # # # #     return face_images
# # # # #
# # # # # def display_faces(face_images):
# # # # #     """מציג את הפנים שזוהו."""
# # # # #     plt.figure(figsize=(10, 10))
# # # # #     for i, face in enumerate(face_images):
# # # # #         plt.subplot(1, len(face_images), i + 1)
# # # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # # #         plt.axis('off')
# # # # #     plt.show()
# # # # #
# # # # # def resize_image(image, scale_percent):
# # # # #     """משנה את גודל התמונה לפי אחוז נתון."""
# # # # #     width = int(image.shape[1] * scale_percent / 100)
# # # # #     height = int(image.shape[0] * scale_percent / 100)
# # # # #     dim = (width, height)
# # # # #     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# # # # #
# # # # # def recognize_faces(image_b, known_faces, threshold=0.5):  # סף נמוך יותר
# # # # #     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה רשימה של אינדקסים של הפנים המזוהות."""
# # # # #     image = resize_image(image_b, 20)
# # # # #     face_locations = detect_faces(image)
# # # # #     recognized_faces_indices = []
# # # # #
# # # # #     for face_location in face_locations:
# # # # #         (top, right, bottom, left) = face_location
# # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # # # #
# # # # #         # בודק אם המרחקים קטנים מהסף
# # # # #         match_index = np.argmin(distances)
# # # # #         if distances[match_index] < threshold:
# # # # #             recognized_faces_indices.append(match_index)
# # # # #         else:
# # # # #             recognized_faces_indices.append(-1)
# # # # #
# # # # #         # מצייר מלבן סביב הפנים ומוסיף את האינדקס
# # # # #         cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
# # # # #         cv2.putText(image, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # # # #
# # # # #     return image, recognized_faces_indices
# # # # # #
# # # # # # def face_exists(new_encoding):
# # # # # #     """בודק אם קידוד הפנים כבר קיים במסד הנתונים."""
# # # # # #     existing_faces = session.query(Face).all()  # מקבל את כל הפנים הקיימות
# # # # # #     for face in existing_faces:
# # # # # #         distance = face_recognition.face_distance([face.encoding], new_encoding)
# # # # # #         if distance < 0.6:  # סף השוואה
# # # # # #             return True
# # # # # #     return False
# # # # #
# # # # #
# # # # # # פונקציה שמחזירה את כל הפנים השמורות
# # # # # def get_all_faces(session):
# # # # #     existing_faces = session.query(Face).all()
# # # # #     face_data = []
# # # # #
# # # # #     for face in existing_faces:
# # # # #         try:
# # # # #             # קבלת הקידוד כ-binary מהעמודה
# # # # #             face_encoding_bytes = face.encoding
# # # # #
# # # # #             # פרוק הקידוד
# # # # #             face_encoding = pickle.loads(face_encoding_bytes)
# # # # #             face_data.append((face.id, face_encoding))  # שמירה של ID וקידוד
# # # # #         except Exception as e:
# # # # #             print(f"Error unpickling face encoding: {e}")
# # # # #
# # # # #     return face_data
# # # # #
# # # # #
# # # # # # פונקציה שבודקת אם הפנים קיימות ומחזירה את ה-ID של הפנים שנמצאו
# # # # # def face_exists(session, new_encoding, threshold=0.6):
# # # # #     face_data = get_all_faces(session)
# # # # #
# # # # #     closest_id = -1
# # # # #     closest_distance = float('inf')  # מתחילים עם מרחק אינסופי
# # # # #
# # # # #     for face_id, face_encoding in face_data:
# # # # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # # # #
# # # # #         # אם המרחק קטן מהסף, נעדכן את ה-ID והמרחק הכי קרוב
# # # # #         if distance < threshold and distance < closest_distance:
# # # # #             closest_distance = distance
# # # # #             closest_id = face_id
# # # # #
# # # # #     return closest_id
# # # # #
# # # # #
# # # # # #
# # # # # # def save_image_to_db(image_path, face_encodings, face_images):
# # # # # #     # קריאת התמונה מקובץ
# # # # # #     with open(image_path, 'rb') as file:
# # # # # #         image_data = file.read()
# # # # # #
# # # # # #     # יצירת מופע של המודל
# # # # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # # # #     session.add(new_image)
# # # # # #     session.commit()  # שמירת התמונה כדי לקבל את ה-ID שלה
# # # # # #
# # # # # #     # שמירת הפנים המזוהות
# # # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # # #         if not face_exists(encoding):  # בודק אם הפנים לא קיימות
# # # # # #             # שמירת התמונה החתוכה של הפנים
# # # # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # # # #             face_image_data = buffer.tobytes()
# # # # # #
# # # # # #             new_face = Face(encoding=encoding, face_image_data=face_image_data)
# # # # # #             session.add(new_face)
# # # # # #             session.commit()  # שמירה כדי לקבל את ה-ID של הפנים
# # # # # #
# # # # # #             # יצירת קשר בין התמונה לפנים
# # # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # # #             session.add(link)
# # # # # #
# # # # # #     session.commit()
# # # # # #     print("Image and new faces uploaded successfully.")
# # # # #
# # # # # def save_image_to_db(image_path, face_encodings, face_images):
# # # # #     with open(image_path, 'rb') as file:
# # # # #         image_data = file.read()
# # # # #
# # # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # # #     session.add(new_image)
# # # # #     session.commit()
# # # # #
# # # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # # #         if not face_exists(encoding):  # בודק אם הפנים לא קיימות
# # # # #             # שמירת התמונה החתוכה של הפנים
# # # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # # #             face_image_data = buffer.tobytes()
# # # # #
# # # # #             # שמירה של הקידוד כ-Pickle
# # # # #             encoded_face = pickle.dumps(encoding)
# # # # #
# # # # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # # # #             session.add(new_face)
# # # # #             session.commit()
# # # # #
# # # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # # #             session.add(link)
# # # # #
# # # # #     session.commit()
# # # # #     print("Image and new faces uploaded successfully.")
# # # # #
# # # # #
# # # # # def main():
# # # # #     # טוען את התמונה המקורית
# # # # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# # # # #     image = load_image(original_image_path)
# # # # #
# # # # #     # מזהה את הפנים בתמונה
# # # # #     face_locations = detect_faces(image)
# # # # #     face_images = extract_faces(image, face_locations)
# # # # #
# # # # #     # מציג את הפנים
# # # # #     display_faces(face_images)
# # # # #
# # # # #     # רשימה לשמירת הפנים המזוהות
# # # # #     known_faces = []
# # # # #     face_encodings = []  # רשימה לשמירת קידוד הפנים
# # # # #     for face_location in face_locations:
# # # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # # #         known_faces.append(face_encoding)
# # # # #         face_encodings.append(face_encoding)  # שמירת הקידוד
# # # # #
# # # # #     # שמירת התמונה והפנים במסד הנתונים
# # # # #     save_image_to_db(original_image_path, face_encodings, face_images)
# # # # #
# # # # #     # טוען תמונה חדשה להשוואה
# # # # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# # # # #     new_image = load_image(new_image_path)
# # # # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # # # #
# # # # #     # מציג את התמונה עם הפנים המזוהות
# # # # #     cv2.imshow("Recognized Faces", recognized_image)
# # # # #     cv2.waitKey(0)
# # # # #     cv2.destroyAllWindows()
# # # # #
# # # # #     # הצגת האינדקסים המזוהות
# # # # #     print("Recognized Indices: ", recognized_faces_indices)
# # # # #
# # # # # if __name__ == "__main__":
# # # # #     main()
# # # # #
# # # # # # סגירת הסשן
# # # # # session.close()
# # # # import pickle
# # # # import cv2
# # # # import numpy as np
# # # # import face_recognition
# # # # import matplotlib.pyplot as plt
# # # # from sqlalchemy.dialects.mysql import LONGBLOB
# # # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # # #
# # # # # הגדרת בסיס המודל
# # # # Base = declarative_base()
# # # #
# # # # class Image(Base):
# # # #     __tablename__ = 'images'
# # # #
# # # #     id = Column(Integer, primary_key=True)
# # # #     image_data = Column(LONGBLOB, nullable=False)
# # # #     image_name = Column(String(255), nullable=False)
# # # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # # #
# # # #
# # # # class Face(Base):
# # # #     __tablename__ = 'faces'
# # # #
# # # #     id = Column(Integer, primary_key=True)
# # # #     encoding = Column(LONGBLOB, nullable=False)  # קידוד הפנים
# # # #     face_image_data = Column(LONGBLOB, nullable=False)  # תמונה חתוכה של הפנים
# # # #     images = relationship("ImageFaceLink", back_populates="face")  # קשר עם טבלת הקישור
# # # #
# # # # class ImageFaceLink(Base):
# # # #     __tablename__ = 'image_face_link'
# # # #
# # # #     id = Column(Integer, primary_key=True)
# # # #     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
# # # #     face_id = Column(Integer, ForeignKey('faces.id'), nullable=False)
# # # #     image = relationship("Image", back_populates="face_links")
# # # #     face = relationship("Face", back_populates="images")
# # # #
# # # # # חיבור למסד הנתונים
# # # # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# # # # Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
# # # #
# # # # # יצירת סשן
# # # # Session = sessionmaker(bind=engine)
# # # # session = Session()
# # # #
# # # # def load_image(image_path):
# # # #     """טוען תמונה מכתובת נתונה."""
# # # #     image = cv2.imread(image_path)
# # # #     if image is None:
# # # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # # #     return image
# # # #
# # # # def detect_faces(image):
# # # #     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
# # # #     return face_recognition.face_locations(image, model="hog")
# # # #
# # # # def extract_faces(image, face_locations):
# # # #     """חותך את הפנים מהתמונה ומחזיר רשימה של תמונות הפנים."""
# # # #     face_images = []
# # # #     for (top, right, bottom, left) in face_locations:
# # # #         face_image = image[top:bottom, left:right]
# # # #         face_images.append(face_image)
# # # #     return face_images
# # # #
# # # # def display_faces(face_images):
# # # #     """מציג את הפנים שזוהו."""
# # # #     plt.figure(figsize=(10, 10))
# # # #     for i, face in enumerate(face_images):
# # # #         plt.subplot(1, len(face_images), i + 1)
# # # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # # #         plt.axis('off')
# # # #     plt.show()
# # # #
# # # # def resize_image(image, scale_percent):
# # # #     """משנה את גודל התמונה לפי אחוז נתון."""
# # # #     width = int(image.shape[1] * scale_percent / 100)
# # # #     height = int(image.shape[0] * scale_percent / 100)
# # # #     dim = (width, height)
# # # #     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# # # #
# # # # def recognize_faces(image_b, known_faces, threshold=0.5):
# # # #     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה רשימה של אינדקסים של הפנים המזוהות."""
# # # #     image = resize_image(image_b, 20)
# # # #     face_locations = detect_faces(image)
# # # #     recognized_faces_indices = []
# # # #
# # # #     for face_location in face_locations:
# # # #         (top, right, bottom, left) = face_location
# # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # # #
# # # #         match_index = np.argmin(distances)
# # # #         if distances[match_index] < threshold:
# # # #             recognized_faces_indices.append(match_index)
# # # #         else:
# # # #             recognized_faces_indices.append(-1)
# # # #
# # # #         cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
# # # #         cv2.putText(image, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # # #
# # # #     return image, recognized_faces_indices
# # # #
# # # # def get_all_faces(session):
# # # #     existing_faces = session.query(Face).all()
# # # #     face_data = []
# # # #
# # # #     for face in existing_faces:
# # # #         try:
# # # #             face_encoding_bytes = face.encoding
# # # #             face_encoding = pickle.loads(face_encoding_bytes)
# # # #             face_data.append((face.id, face_encoding))
# # # #         except Exception as e:
# # # #             print(f"Error unpickling face encoding: {e}")
# # # #
# # # #     return face_data
# # # #
# # # # def face_exists(session, new_encoding, threshold=0.6):
# # # #     face_data = get_all_faces(session)
# # # #
# # # #     closest_id = -1
# # # #     closest_distance = float('inf')
# # # #
# # # #     for face_id, face_encoding in face_data:
# # # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # # #
# # # #         if distance < threshold and distance < closest_distance:
# # # #             closest_distance = distance
# # # #             closest_id = face_id
# # # #
# # # #     return closest_id
# # # #
# # # # def save_image_to_db(image_path, face_encodings, face_images):
# # # #     with open(image_path, 'rb') as file:
# # # #         image_data = file.read()
# # # #
# # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # #     session.add(new_image)
# # # #     session.commit()
# # # #
# # # #     for encoding, face_image in zip(face_encodings, face_images):
# # # #         if face_exists(session, encoding) == -1:  # בודק אם הפנים לא קיימות
# # # #             _, buffer = cv2.imencode('.jpg', face_image)
# # # #             face_image_data = buffer.tobytes()
# # # #             encoded_face = pickle.dumps(encoding)
# # # #
# # # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # # #             session.add(new_face)
# # # #             session.commit()
# # # #
# # # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # # #             session.add(link)
# # # #
# # # #     session.commit()
# # # #     print("Image and new faces uploaded successfully.")
# # # #
# # # # def main():
# # # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# # # #     image = load_image(original_image_path)
# # # #
# # # #     face_locations = detect_faces(image)
# # # #     face_images = extract_faces(image, face_locations)
# # # #
# # # #     display_faces(face_images)
# # # #
# # # #     known_faces = []
# # # #     face_encodings = []
# # # #     for face_location in face_locations:
# # # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # # #         known_faces.append(face_encoding)
# # # #         face_encodings.append(face_encoding)
# # # #
# # # #     save_image_to_db(original_image_path, face_encodings, face_images)
# # # #
# # # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# # # #     new_image = load_image(new_image_path)
# # # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # # #
# # # #     cv2.imshow("Recognized Faces", recognized_image)
# # # #     cv2.waitKey(0)
# # # #     cv2.destroyAllWindows()
# # # #
# # # #     print("Recognized Indices: ", recognized_faces_indices)
# # # #
# # # # if __name__ == "__main__":
# # # #     main()
# # # #
# # # # session.close()
# # #
# # #
# # #
# # # import pickle
# # # import cv2
# # # import numpy as np
# # # import face_recognition
# # # import matplotlib.pyplot as plt
# # # from sqlalchemy.dialects.mysql import LONGBLOB
# # # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# # #
# # # # הגדרת בסיס המודל
# # # Base = declarative_base()
# # #
# # # class Image(Base):
# # #     __tablename__ = 'images'
# # #     id = Column(Integer, primary_key=True)
# # #     image_data = Column(LONGBLOB, nullable=False)
# # #     image_name = Column(String(255), nullable=False)
# # #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# # #
# # # class Face(Base):
# # #     __tablename__ = 'faces'
# # #     id = Column(Integer, primary_key=True)
# # #     encoding = Column(LONGBLOB, nullable=False)
# # #     face_image_data = Column(LONGBLOB, nullable=False)
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
# # #     engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# # #     Base.metadata.create_all(engine)
# # #     return sessionmaker(bind=engine)()
# # #
# # # # פונקציות לעיבוד תמונה
# # # def load_image(image_path):
# # #     image = cv2.imread(image_path)
# # #     if image is None:
# # #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# # #     return image
# # #
# # # def detect_faces(image):
# # #     return face_recognition.face_locations(image, model="hog")
# # #
# # # def extract_faces(image, face_locations):
# # #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# # #
# # # def display_faces(face_images):
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
# # # def recognize_faces(image, known_faces, threshold=0.5):
# # #     image_resized = resize_image(image, 20)
# # #     face_locations = detect_faces(image_resized)
# # #     recognized_faces_indices = []
# # #
# # #     for face_location in face_locations:
# # #         (top, right, bottom, left) = face_location
# # #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # #
# # #         match_index = np.argmin(distances)
# # #         recognized_faces_indices.append(match_index if distances[match_index] < threshold else -1)
# # #
# # #         cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)
# # #         cv2.putText(image_resized, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # #
# # #     return image_resized, recognized_faces_indices
# # #
# # # # פונקציות למסד נתונים
# # # def get_all_faces(session):
# # #     existing_faces = session.query(Face).all()
# # #     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
# # #
# # # def face_exists(session, new_encoding, threshold=0.6):
# # #     face_data = get_all_faces(session)
# # #     closest_id = -1
# # #     closest_distance = float('inf')
# # #
# # #     for face_id, face_encoding in face_data:
# # #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# # #         if distance < threshold and distance < closest_distance:
# # #             closest_distance = distance
# # #             closest_id = face_id
# # #
# # #     return closest_id
# # #
# # # def save_image_to_db(session, image_path, face_encodings, face_images):
# # #     with open(image_path, 'rb') as file:
# # #         image_data = file.read()
# # #
# # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # #     session.add(new_image)
# # #     session.commit()
# # #
# # #     for encoding, face_image in zip(face_encodings, face_images):
# # #         if face_exists(session, encoding) == -1:
# # #             _, buffer = cv2.imencode('.jpg', face_image)
# # #             face_image_data = buffer.tobytes()
# # #             encoded_face = pickle.dumps(encoding)
# # #
# # #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# # #             session.add(new_face)
# # #             session.commit()
# # #
# # #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# # #             session.add(link)
# # #
# # #     session.commit()
# # #     print("Image and new faces uploaded successfully.")
# # #
# # # def main():
# # #     session = create_database_connection()
# # #
# # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# # #     image = load_image(original_image_path)
# # #     face_locations = detect_faces(image)
# # #     face_images = extract_faces(image, face_locations)
# # #
# # #     display_faces(face_images)
# # #
# # #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# # #     save_image_to_db(session, original_image_path, known_faces, face_images)
# # #
# # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# # #     new_image = load_image(new_image_path)
# # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # #
# # #     cv2.imshow("Recognized Faces", recognized_image)
# # #     cv2.waitKey(0)
# # #     cv2.destroyAllWindows()
# # #     print("Recognized Indices: ", recognized_faces_indices)
# # #
# # #     session.close()
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# #
# # import pickle
# # import cv2
# # import numpy as np
# # import face_recognition
# # import matplotlib.pyplot as plt
# # from sqlalchemy.dialects.mysql import LONGBLOB
# # from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# # from sqlalchemy.orm import sessionmaker, declarative_base, relationship
# #
# # # הגדרת בסיס המודל
# # Base = declarative_base()
# #
# # class Image(Base):
# #     __tablename__ = 'images'
# #     id = Column(Integer, primary_key=True)
# #     image_data = Column(LONGBLOB, nullable=False)
# #     image_name = Column(String(255), nullable=False)
# #     face_links = relationship("ImageFaceLink", back_populates="image", cascade="all, delete-orphan")
# #
# # class Face(Base):
# #     __tablename__ = 'faces'
# #     id = Column(Integer, primary_key=True)
# #     encoding = Column(LONGBLOB, nullable=False)
# #     face_image_data = Column(LONGBLOB, nullable=False)
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
# #     engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image2")
# #     Base.metadata.create_all(engine)
# #     return sessionmaker(bind=engine)()
# #
# # # פונקציות לעיבוד תמונה
# # def load_image(image_path):
# #     image = cv2.imread(image_path)
# #     if image is None:
# #         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
# #     return image
# #
# # def detect_faces(image):
# #     return face_recognition.face_locations(image, model="hog")
# #
# # def extract_faces(image, face_locations):
# #     return [image[top:bottom, left:right] for (top, right, bottom, left) in face_locations]
# #
# # def display_faces(face_images):
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
# # #
# # # # פונקציות לזיהוי פנים
# # # def recognize_faces(image, known_faces, threshold=0.5):
# # #     image_resized = resize_image(image, 20)
# # #     face_locations = detect_faces(image_resized)
# # #     recognized_faces_indices = []
# # #
# # #     for face_location in face_locations:
# # #         (top, right, bottom, left) = face_location
# # #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # #
# # #         match_index = np.argmin(distances)
# # #         recognized_faces_indices.append(match_index if distances[match_index] < threshold else -1)
# # #
# # #         cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)
# # #         cv2.putText(image_resized, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # #
# # #     return image_resized, recognized_faces_indices
# #
# #
# # def recognize_faces(image, session, known_faces, threshold=0.5):
# #     image_resized = resize_image(image, 20)
# #     face_locations = detect_faces(image_resized)
# #     recognized_faces_indices = []
# #
# #     for face_location in face_locations:
# #         (top, right, bottom, left) = face_location
# #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# #
# #         # שימוש בפונקציה face_exists כדי לבדוק אם הפנים כבר קיימות
# #         existing_face_id = face_exists(session, face_encoding, threshold)
# #         recognized_faces_indices.append(existing_face_id)
# #
# #         # אם הפנים לא קיימות, ניתן להוסיף קוד לצייר מלבן סביב הפנים
# #         if existing_face_id == -1:
# #             cv2.rectangle(image_resized, (left, top), (right, bottom), (0, 255, 0), 2)  # צבע ירוק עבור פנים חדשות
# #         else:
# #             cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)  # צבע אדום עבור פנים מוכרות
# #             cv2.putText(image_resized, f"ID: {existing_face_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
# #                         (255, 0, 0), 2)
# #
# #     return image_resized, recognized_faces_indices
# #
# #
# # # פונקציות למסד נתונים
# # def get_all_faces(session):
# #     existing_faces = session.query(Face).all()
# #     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
# #
# # def face_exists(session, new_encoding, threshold=0.6):
# #     face_data = get_all_faces(session)
# #     closest_id = -1
# #     closest_distance = float('inf')
# #
# #     for face_id, face_encoding in face_data:
# #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# #         if distance < threshold and distance < closest_distance:
# #             closest_distance = distance
# #             closest_id = face_id
# #
# #     return closest_id
# #
# # def save_image_to_db(session, image_path, face_encodings, face_images):
# #     with open(image_path, 'rb') as file:
# #         image_data = file.read()
# #
# #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# #     session.add(new_image)
# #     session.commit()
# #
# #     for encoding, face_image in zip(face_encodings, face_images):
# #         if face_exists(session, encoding) == -1:
# #             _, buffer = cv2.imencode('.jpg', face_image)
# #             face_image_data = buffer.tobytes()
# #             encoded_face = pickle.dumps(encoding)
# #
# #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# #             session.add(new_face)
# #             session.commit()
# #
# #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# #             session.add(link)
# #
# #     session.commit()
# #     print("Image and new faces uploaded successfully.")
# #
# # def main():
# #     session = create_database_connection()
# #
# #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# #     image = load_image(original_image_path)
# #     face_locations = detect_faces(image)
# #     face_images = extract_faces(image, face_locations)
# #
# #     display_faces(face_images)
# #
# #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# #     save_image_to_db(session, original_image_path, known_faces, face_images)
# #
# #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# #     new_image = load_image(new_image_path)
# #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# #
# #     cv2.imshow("Recognized Faces", recognized_image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #     print("Recognized Indices: ", recognized_faces_indices)
# #
# #     session.close()
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
# # # פונקציות למסד נתונים
# # def get_all_faces(session):
# #     existing_faces = session.query(Face).all()
# #     return [(face.id, pickle.loads(face.encoding)) for face in existing_faces]
# #
# # def face_exists(session, new_encoding, threshold=0.6):
# #     face_data = get_all_faces(session)
# #     closest_id = -1
# #     closest_distance = float('inf')
# #
# #     for face_id, face_encoding in face_data:
# #         distance = face_recognition.face_distance([face_encoding], new_encoding)[0]
# #         if distance < threshold and distance < closest_distance:
# #             closest_distance = distance
# #             closest_id = face_id
# #
# #     return closest_id
# #
# # def save_image_to_db(session, image_path, face_encodings, face_images):
# #     with open(image_path, 'rb') as file:
# #         image_data = file.read()
# #
# #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# #     session.add(new_image)
# #     session.commit()
# #
# #     for encoding, face_image in zip(face_encodings, face_images):
# #         if face_exists(session, encoding) == -1:
# #             _, buffer = cv2.imencode('.jpg', face_image)
# #             face_image_data = buffer.tobytes()
# #             encoded_face = pickle.dumps(encoding)
# #
# #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# #             session.add(new_face)
# #             session.commit()
# #
# #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# #             session.add(link)
# #
# #     session.commit()
# #     print("Image and new faces uploaded successfully.")
# #
# # def main():
# #     session = create_database_connection()
# #
# #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# #     image = load_image(original_image_path)
# #     face_locations = detect_faces(image)
# #     face_images = extract_faces(image, face_locations)
# #
# #     display_faces(face_images)
# #
# #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# #     save_image_to_db(session, original_image_path, known_faces, face_images)
# #
# #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# #     new_image = load_image(new_image_path)
# #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# #
# #     cv2.imshow("Recognized Faces", recognized_image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #     print("Recognized Indices: ", recognized_faces_indices)
# #
# #     session.close()
# #
# # if __name__ == "__main__":
# #     main()
# #
# #
#
#
# import pickle
# import cv2
# import numpy as np
# import face_recognition
# import matplotlib.pyplot as plt
# from sqlalchemy.dialects.mysql import LONGBLOB
# from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
# from sqlalchemy.orm import sessionmaker, declarative_base, relationship
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
#     engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image3")
#     Base.metadata.create_all(engine)
#     return sessionmaker(bind=engine)()
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
#
# #
# # # פונקציות לזיהוי פנים
# # def recognize_faces(image, known_faces, threshold=0.5):
# #     image_resized = resize_image(image, 20)
# #     face_locations = detect_faces(image_resized)
# #     recognized_faces_indices = []
# #
# #     for face_location in face_locations:
# #         (top, right, bottom, left) = face_location
# #         face_encoding = face_recognition.face_encodings(image_resized, [face_location])[0]
# #         distances = face_recognition.face_distance(known_faces, face_encoding)
# #
# #         match_index = np.argmin(distances)
# #         recognized_faces_indices.append(match_index if distances[match_index] < threshold else -1)
# #
# #         cv2.rectangle(image_resized, (left, top), (right, bottom), (255, 0, 0), 2)
# #         cv2.putText(image_resized, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# #
# #     return image_resized, recognized_faces_indices
#
#
#
# # פונקציות למסד נתונים
#
# def recognize_faces(image, session, known_faces, threshold=0.6):
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
#
#
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
# #
# # def save_image_to_db(session, image_path, face_encodings, face_images):
# #     with open(image_path, 'rb') as file:
# #         image_data = file.read()
# #
# #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# #     session.add(new_image)
# #     session.commit()
# #
# #     for encoding, face_image in zip(face_encodings, face_images):
# #         if face_exists(session, encoding) == -1:
# #             _, buffer = cv2.imencode('.jpg', face_image)
# #             face_image_data = buffer.tobytes()
# #             encoded_face = pickle.dumps(encoding)
# #
# #             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
# #             session.add(new_face)
# #             session.commit()
# #
# #             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
# #             session.add(link)
# #
# #     session.commit()
# #     print("Image and new faces uploaded successfully.")
# # #
# # def main():
# #     session = create_database_connection()
# #
# #     original_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (124).jpg"
# #     image = load_image(original_image_path)
# #     face_locations = detect_faces(image)
# #     face_images = extract_faces(image, face_locations)
# #
# #     display_faces(face_images)
# #
# #     known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
# #     save_image_to_db(session, original_image_path, known_faces, face_images)
# #
# #     new_image_path = r"C:\Users\user1\Pictures\for-practic\2_ (149).jpg"
# #     new_image = load_image(new_image_path)
# #     recognized_image, recognized_faces_indices = recognize_faces(new_image,session, known_faces)
# #
# #     cv2.imshow("Recognized Faces", recognized_image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #     print("Recognized Indices: ", recognized_faces_indices)
# #
# #     session.close()
# #
# # if __name__ == "__main__":
# #     main()
#
#
# def display_faces_with_ids(face_images, face_ids):
#     plt.figure(figsize=(10, 10))
#     for i, (face, face_id) in enumerate(zip(face_images, face_ids)):
#         plt.subplot(1, len(face_images), i + 1)
#         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#         plt.title(f"ID: {face_id}")
#         plt.axis('off')
#     plt.show()
#
# def get_all_images(session):
#     images = session.query(Image).all()
#     for image in images:
#         print(f"Image ID: {image.id}, Name: {image.image_name}")
# #
# # def get_all_faces_with_images(session):
# #     faces = session.query(Face).all()
# #     face_images = []
# #     face_ids = []
# #     for face in faces:
# #         face_ids.append(face.id)
# #         face_images.append(pickle.loads(face.face_image_data))
# #     display_faces_with_ids(face_images, face_ids)
# #
# # def display_image_by_face_id(session, face_id):
# #     face = session.query(Face).filter(Face.id == face_id).first()
# #     if face:
# #         face_image = pickle.loads(face.face_image_data)
# #         plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
# #         plt.title(f"Face ID: {face_id}")
# #         plt.axis('off')
# #         plt.show()
# #     else:
# #         print("Face not found.")
# def save_image_to_db(session, image_path, face_encodings, face_images):
#     with open(image_path, 'rb') as file:
#         image_data = file.read()
#
#     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
#     session.add(new_image)
#     session.commit()
#
#     for encoding, face_image in zip(face_encodings, face_images):
#         if face_exists(session, encoding) == -1:
#             _, buffer = cv2.imencode('.jpg', face_image)
#             face_image_data = buffer.tobytes()
#             encoded_face = pickle.dumps(encoding)  # קידוד ה-encoding של הפנים
#
#             new_face = Face(encoding=encoded_face, face_image_data=face_image_data)
#             session.add(new_face)
#             session.commit()
#
#             link = ImageFaceLink(image_id=new_image.id, face_id=new_face.id)
#             session.add(link)
#
#     session.commit()
#     print("Image and new faces uploaded successfully.")
#
# def get_all_faces_with_images(session):
#     faces = session.query(Face).all()
#     face_images = []
#     face_ids = []
#     for face in faces:
#         face_ids.append(face.id)
#         try:
#             face_images.append(pickle.loads(face.face_image_data))  # טיפול בשגיאות
#         except Exception as e:
#             print(f"Error loading face image data for face ID {face.id}: {e}")
#     display_faces_with_ids(face_images, face_ids)
#
# def display_image_by_face_id(session, face_id):
#     face = session.query(Face).filter(Face.id == face_id).first()
#     if face:
#         try:
#             face_image = pickle.loads(face.face_image_data)  # טיפול בשגיאות
#             plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
#             plt.title(f"Face ID: {face_id}")
#             plt.axis('off')
#             plt.show()
#         except Exception as e:
#             print(f"Error loading face image data for face ID {face_id}: {e}")
#     else:
#         print("Face not found.")
#
# def main():
#     session = create_database_connection()
#
#     while True:
#         print("בחר פעולה:")
#         print("1. הוספת תמונה חדשה")
#         print("2. קבלת כל התמונות")
#         print("3. קבלת כל הפנים")
#         print("4. הצגת תמונות לפי מזהה פנים")
#         print("5. יציאה")
#
#         choice = input("הכנס את מספר הבחירה שלך: \n")
#
#         if choice == '1':
#             original_image_path = input("הכנס את נתיב התמונה להוספה: ")
#             image = load_image(original_image_path)
#             face_locations = detect_faces(image)
#             face_images = extract_faces(image, face_locations)
#             display_faces(face_images)
#             known_faces = [face_recognition.face_encodings(image, [face_location])[0] for face_location in face_locations]
#             save_image_to_db(session, original_image_path, known_faces, face_images)
#
#         elif choice == '2':
#             get_all_images(session)
#
#         elif choice == '3':
#             get_all_faces_with_images(session)
#
#         elif choice == '4':
#             face_id = int(input("הכנס את מזהה הפנים: "))
#             display_image_by_face_id(session, face_id)
#
#         elif choice == '5':
#             break
#
#         else:
#             print("בחירה לא חוקית. אנא נסה שוב.")
#
#     session.close()
#
# if __name__ == "__main__":
#     main()
