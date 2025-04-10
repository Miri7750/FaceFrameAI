# # # import cv2
# # # import numpy as np
# # # import face_recognition
# # # import matplotlib.pyplot as plt
# # #
# # # def load_image(image_path):
# # #     """טוען תמונה מכתובת נתונה."""
# # #     return cv2.imread(image_path)
# # #
# # # def detect_faces(image):
# # #     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
# # #     return face_recognition.face_locations(image)
# # #
# # # def extract_faces(image, face_locations):
# # #     """חותך את הפנים מהתמונה ומחזיר רשימה של תמונות הפנים."""
# # #     face_images = []
# # #     for (top, right, bottom, left) in face_locations:
# # #         face_image = image[top:bottom, left:right]
# # #         face_images.append(face_image)
# # #     return face_images
# # #
# # # def display_faces(face_images):
# # #     """מציג את הפנים שזוהו."""
# # #     plt.figure(figsize=(10, 10))
# # #     for i, face in enumerate(face_images):
# # #         plt.subplot(1, len(face_images), i + 1)
# # #         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# # #         plt.axis('off')
# # #     plt.show()
# # #
# # # def resize_image(image, scale_percent):
# # #     """משנה את גודל התמונה לפי אחוז נתון."""
# # #     width = int(image.shape[1] * scale_percent / 100)
# # #     height = int(image.shape[0] * scale_percent / 100)
# # #     dim = (width, height)
# # #     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# # #
# # # def recognize_faces(image_b, known_faces, threshold=0.6):
# # #     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה רשימה של אינדקסים של הפנים המזוהות."""
# # #     image = resize_image(image_b, 20)
# # #     face_locations = detect_faces(image)
# # #     recognized_faces_indices = []
# # #
# # #     for face_location in face_locations:
# # #         (top, right, bottom, left) = face_location
# # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # #         distances = face_recognition.face_distance(known_faces, face_encoding)
# # #
# # #         # בודק אם המרחקים קטנים מהסף
# # #         match_index = np.argmin(distances)
# # #         if distances[match_index] < threshold:
# # #             recognized_faces_indices.append(match_index)
# # #         else:
# # #             recognized_faces_indices.append(-1)
# # #
# # #         # מצייר מלבן סביב הפנים ומוסיף את האינדקס
# # #         cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
# # #         cv2.putText(image, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# # #
# # #     return image, recognized_faces_indices
# # #
# # # def save_known_faces(known_faces, filename='known_faces.npy'):
# # #     """שומר את הפנים המוכרות לקובץ."""
# # #     np.save(filename, known_faces)
# # #
# # # def load_known_faces(filename='known_faces.npy'):
# # #     """טוען את הפנים המוכרות מקובץ."""
# # #     return np.load(filename, allow_pickle=True)
# # #
# # # def main():
# # #     # טוען את התמונה המקורית
# # #     original_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7361.JPG"
# # #     image = load_image(original_image_path)
# # #
# # #     # מזהה את הפנים בתמונה
# # #     face_locations = detect_faces(image)
# # #     face_images = extract_faces(image, face_locations)
# # #
# # #     # מציג את הפנים
# # #     display_faces(face_images)
# # #
# # #     # שומר את הפנים המוכרות
# # #     known_faces = []  # רשימה לשמירת הפנים המוכרות
# # #     for face_location in face_locations:
# # #         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
# # #         known_faces.append(face_encoding)
# # #
# # #     # טוען תמונה חדשה להשוואה
# # #     new_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7339.JPG"
# # #     new_image = load_image(new_image_path)
# # #     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
# # #
# # #     # מציג את התמונה עם הפנים המזוהות
# # #     cv2.imshow("Recognized Faces", recognized_image)
# # #     cv2.waitKey(0)
# # #     cv2.destroyAllWindows()
# # #
# # #     # הצגת האינדקסים המזוהות
# # #     print("Recognized Indices: ", recognized_faces_indices)
# # #
# # # if __name__ == "__main__":
# # #     main()
# #
# #
# # import cv2
# # import numpy as np
# # import face_recognition
# #
# #
# # def load_image(image_path):
# #     """טוען תמונה מכתובת נתונה."""
# #     return cv2.imread(image_path)
# #
# #
# # def detect_faces(image):
# #     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
# #     return face_recognition.face_locations(image)
# #
# #
# # def recognize_faces(image_b, known_faces):
# #     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה את הפנים עם הקרבה הכי גדולה."""
# #     face_locations = detect_faces(image_b)
# #     recognized_faces = []
# #
# #     for face_location in face_locations:
# #         (top, right, bottom, left) = face_location
# #         face_encoding = face_recognition.face_encodings(image_b, [face_location])[0]
# #
# #         # חישוב המרחקים בין הפנים המוכרות לפנים המזוהות
# #         distances = face_recognition.face_distance(known_faces, face_encoding)
# #
# #         # מציאת האינדקס של הפנים עם המרחק הקטן ביותר
# #         min_index = np.argmin(distances)
# #         min_distance = distances[min_index]
# #
# #         # הוספת התוצאה לרשימה
# #         recognized_faces.append((min_index, min_distance))
# #
# #         # צייר מלבן סביב הפנים
# #         cv2.rectangle(image_b, (left, top), (right, bottom), (255, 0, 0), 2)
# #         cv2.putText(image_b, f"ID: {min_index}, Dist: {min_distance:.2f}", (left, top - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
# #
# #     return image_b, recognized_faces
# #
# #
# # def main():
# #     # טוען את התמונה המקורית
# #     original_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_6354.JPG"
# #     image = load_image(original_image_path)
# #
# #     # קידודים מוכרים
# #     known_faces = []  # כאן יש להוסיף את הקידודים של הפנים המוכרות
# #
# #     # זיהוי הפנים בתמונה
# #     recognized_image, recognized_faces = recognize_faces(image, known_faces)
# #
# #     # מציג את התמונה עם הפנים המזוהות
# #     cv2.imshow("Recognized Faces", recognized_image)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
# #
# #     # הצגת התוצאות
# #     for index, distance in recognized_faces:
# #         print(f"Recognized ID: {index}, Distance: {distance:.2f}")
# #
# #
# # if __name__ == "__main__":
# #     main()
#
#
#
# import cv2
# import numpy as np
# import face_recognition
# import matplotlib.pyplot as plt
#
# def load_image(image_path):
#     """טוען תמונה מכתובת נתונה."""
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Unable to load image at {image_path}. Please check the file path.")
#     return image
#
# def detect_faces(image):
#     """מזהה את מיקומי הפנים בתמונה ומחזיר את המיקומים."""
#     return face_recognition.face_locations(image, model="hog")  # שימוש במודל HOG לזיהוי מדויק יותר
#
# def extract_faces(image, face_locations):
#     """חותך את הפנים מהתמונה ומחזיר רשימה של תמונות הפנים."""
#     face_images = []
#     for (top, right, bottom, left) in face_locations:
#         face_image = image[top:bottom, left:right]
#         face_images.append(face_image)
#     return face_images
#
# def display_faces(face_images):
#     """מציג את הפנים שזוהו."""
#     plt.figure(figsize=(10, 10))
#     for i, face in enumerate(face_images):
#         plt.subplot(1, len(face_images), i + 1)
#         plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#     plt.show()
#
# def resize_image(image, scale_percent):
#     """משנה את גודל התמונה לפי אחוז נתון."""
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#
# def recognize_faces(image_b, known_faces, threshold=0.5):  # סף נמוך יותר
#     """מזהה פנים בתמונה ומשווה עם הפנים המוכרות, מחזירה רשימה של אינדקסים של הפנים המזוהות."""
#     image = resize_image(image_b, 20)
#     face_locations = detect_faces(image)
#     recognized_faces_indices = []
#
#     for face_location in face_locations:
#         (top, right, bottom, left) = face_location
#         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
#         distances = face_recognition.face_distance(known_faces, face_encoding)
#
#         # בודק אם המרחקים קטנים מהסף
#         match_index = np.argmin(distances)
#         if distances[match_index] < threshold:
#             recognized_faces_indices.append(match_index)
#         else:
#             recognized_faces_indices.append(-1)
#
#         # מצייר מלבן סביב הפנים ומוסיף את האינדקס
#         cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
#         cv2.putText(image, f"ID: {recognized_faces_indices[-1]}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     return image, recognized_faces_indices
#
# def save_known_faces(known_faces, filename='known_faces.npy'):
#     """שומר את הפנים המוכרות לקובץ."""
#     np.save(filename, known_faces)
#
# def load_known_faces(filename='known_faces.npy'):
#     """טוען את הפנים המוכרות מקובץ."""
#     return np.load(filename, allow_pickle=True)
#
# def main():
#     # טוען את התמונה המקורית
#     original_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7361small.JPG"
#     image = load_image(original_image_path)
#
#     # מזהה את הפנים בתמונה
#     face_locations = detect_faces(image)
#     face_images = extract_faces(image, face_locations)
#
#     # מציג את הפנים
#     display_faces(face_images)
#
#     # שומר את הפנים המוכרות
#     known_faces = []  # רשימה לשמירת הפנים המוכרות
#     for face_location in face_locations:
#         face_encoding = face_recognition.face_encodings(image, [face_location])[0]
#         known_faces.append(face_encoding)
#
#     # טוען תמונה חדשה להשוואה
#     new_image_path = r"C:\Users\user1\Pictures\for-practic\IMG_7339small.JPG"
#     new_image = load_image(new_image_path)
#     recognized_image, recognized_faces_indices = recognize_faces(new_image, known_faces)
#
#     # מציג את התמונה עם הפנים המזוהות
#     cv2.imshow("Recognized Faces", recognized_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # הצגת האינדקסים המזוהות
#     print("Recognized Indices: ", recognized_faces_indices)
#
# if __name__ == "__main__":
#     main()
