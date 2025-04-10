# # # # # import mysql.connector
# # # # #
# # # # #
# # # # # def save_image_to_db(image_path):
# # # # #     # חיבור למסד הנתונים
# # # # #     conn = mysql.connector.connect(
# # # # #         host='localhost',
# # # # #         user='username',  # החלף עם שם המשתמש שלך
# # # # #         password='password',  # החלף עם הסיסמה שלך
# # # # #         database='database'  # החלף עם שם מסד הנתונים שלך
# # # # #     )
# # # # #
# # # # #     cursor = conn.cursor()
# # # # #
# # # # #     # קריאת התמונה מקובץ
# # # # #     with open(image_path, 'rb') as file:
# # # # #         image_data = file.read()
# # # # #
# # # # #     # הכנסת התמונה למסד הנתונים
# # # # #     sql = "INSERT INTO images (image_data, image_name) VALUES (%s, %s)"
# # # # #     image_name = image_path.split('/')[-1]  # קבלת שם הקובץ
# # # # #     cursor.execute(sql, (image_data, image_name))
# # # # #
# # # # #     conn.commit()
# # # # #     print("Image uploaded successfully.")
# # # # #
# # # # #     # סגירת החיבור
# # # # #     cursor.close()
# # # # #     conn.close()
# # # # #
# # # # #
# # # # # # קריאה לפונקציה עם נתיב התמונה
# # # # # save_image_to_db('path/to/your/image.jpg')
# # # #
# # # #
# # # # from sqlalchemy import create_engine, Column, Integer, String, LONGBLOB,text
# # # # # from sqlalchemy.ext.declarative import declarative_base
# # # # from sqlalchemy.orm import declarative_base
# # # # from sqlalchemy.orm import sessionmaker
# # # #
# # # # # הגדרת בסיס המודל
# # # # Base = declarative_base()
# # # #
# # # #
# # # # # הגדרת המודל
# # # # class Image(Base):
# # # #     __tablename__ = 'images'
# # # #
# # # #     id = Column(Integer, primary_key=True)
# # # #     image_data = Column(LONGBLOB, nullable=False)
# # # #     image_name = Column(String(255), nullable=False)
# # # #
# # # # # חיבור למסד הנתונים
# # # # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/")
# # # #
# # # # # יצירת מסד הנתונים אם הוא לא קיים
# # # # with engine.connect() as connection:
# # # #     connection.execute(text("CREATE DATABASE IF NOT EXISTS try_save_image"))
# # # # # חיבור למסד הנתונים
# # # # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image")
# # # # Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
# # # #
# # # # # יצירת סשן
# # # # Session = sessionmaker(bind=engine)
# # # # session = Session()
# # # #
# # # #
# # # # def save_image_to_db(image_path):
# # # #     # קריאת התמונה מקובץ
# # # #     with open(image_path, 'rb') as file:
# # # #         image_data = file.read()
# # # #
# # # #     # יצירת מופע של המודל
# # # #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# # # #
# # # #     # הוספת התמונה למסד הנתונים
# # # #     session.add(new_image)
# # # #     session.commit()
# # # #     print("Image uploaded successfully.")
# # # #
# # # #
# # # # # קריאה לפונקציה עם נתיב התמונה
# # # # save_image_to_db(r'C:\Users\user1\Pictures\for-practic\IMG_7339.JPG')
# # # #
# # # # # סגירת הסשן
# # # # session.close()
# # #
# from sqlalchemy.dialects.mysql import LONGBLOB
# from sqlalchemy import create_engine, Column, Integer, String
# from sqlalchemy.orm import sessionmaker,declarative_base
#
# # הגדרת בסיס המודל
# Base = declarative_base()
#
# # הגדרת המודל
# class Image(Base):
#     __tablename__ = 'images'
#
#     id = Column(Integer, primary_key=True)
#     image_data = Column(LONGBLOB, nullable=False)  # שינוי ל-LONGBLOB
#     image_name = Column(String(255), nullable=False)
#
# # חיבור למסד הנתונים
# engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image")
# Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
#
# # יצירת סשן
# Session = sessionmaker(bind=engine)
# session = Session()
#
# def save_image_to_db(image_path):
#     # קריאת התמונה מקובץ
#     with open(image_path, 'rb') as file:
#         image_data = file.read()
#
#     # יצירת מופע של המודל
#     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
#
#     # הוספת התמונה למסד הנתונים
#     session.add(new_image)
#     session.commit()
#     print("Image uploaded successfully.")
#
# # קריאה לפונקציה עם נתיב התמונה
# save_image_to_db(r'C:\Users\user1\Pictures\for-practic\IMG_7339.JPG')
#
# # סגירת הסשן
# session.close()
#
#
#
# # from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
# # from sqlalchemy.ext.declarative import declarative_base
# # from sqlalchemy.orm import sessionmaker
# #
# # # הגדרת בסיס המודל
# # Base = declarative_base()
# #
# # # הגדרת המודל
# # class Image(Base):
# #     __tablename__ = 'images'
# #
# #     id = Column(Integer, primary_key=True)
# #     image_data = Column(LargeBinary, nullable=False)  # השתמש ב-LargeBinary
# #     image_name = Column(String(255), nullable=False)
# #
# # # חיבור למסד הנתונים
# # engine = create_engine("mysql+mysqlconnector://root:aA1795aA@localhost/try_save_image")
# # Base.metadata.create_all(engine)  # יצירת הטבלה אם היא לא קיימת
# #
# # # יצירת סשן
# # Session = sessionmaker(bind=engine)
# # session = Session()
# #
# # # פונקציה לשמירת תמונה במסד הנתונים
# # def save_image_to_db(image_path):
# #     with open(image_path, 'rb') as file:
# #         image_data = file.read()
# #
# #     new_image = Image(image_data=image_data, image_name=image_path.split('/')[-1])
# #     session.add(new_image)
# #     session.commit()
# #     print("Image uploaded successfully.")
# #
# # # קריאה לפונקציה עם נתיב התמונה (ודא שהתמונה גדולה מ-5MB)
# # save_image_to_db(r'C:\Users\user1\Pictures\for-practic\IMG_7339.JPG')
# #
# # # סגירת הסשן
# # session.close()
#
# #
# # import mysql.connector
# # from sqlalchemy.dialects.mysql import LONGBLOB
# #
# #
# # def save_image_to_db(image_path):
# #     # חיבור למסד הנתונים
# #     conn = mysql.connector.connect(
# #         host='localhost',
# #         user='root',  # החלף עם שם המשתמש שלך
# #         password='aA1795aA',  # החלף עם הסיסמה שלך
# #         database='try_save_image'  # החלף עם שם מסד הנתונים שלך
# #     )
# #
# #     cursor = conn.cursor()
# #
# #     # קריאת התמונה מקובץ
# #     with open(image_path, 'rb') as file:
# #         image_data = file.read()
# #
# #     # הכנסת התמונה למסד הנתונים
# #     sql = "INSERT INTO images (image_data, image_name) VALUES (%s, %s)"
# #     image_name = image_path.split('/')[-1]  # קבלת שם הקובץ
# #     cursor.execute(sql, (image_data, image_name))
# #
# #     conn.commit()
# #     print("Image uploaded successfully.")
# #
# #     # סגירת החיבור
# #     cursor.close()
# #     conn.close()
# #
# #
# # # קריאה לפונקציה עם נתיב התמונה
# # save_image_to_db(r'C:\Users\user1\Pictures\for-practic\IMG_7339.JPG')
# # from sqlalchemy import BLOB
# # from sqlalchemy.sql.sqltypes import BLOB
# # from sqlalchemy.dialects.mysql import LONGBLOB
