
# Импорт необходимых библиотек
import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder 
import cv2
import time


#Функция, возвращающая площадь пересечения двух прямоугольников (на будущее, для увеличения точности)
def square_intersection(left, right, top, bottom, left2, right2, top2, bottom2):
    x1 = left
    y1 = top
    x2 = right
    y2 = bottom
    x3 = left2
    y3 = right2
    x4 = top2
    y4 = bottom2
    left_final = max(x1, x3)
    top_final = min(y2, y4)
    right_final = min(x2, x4)
    bottom_final = max(y1, y3)
    return ((right_final - left_final) * (top_final - bottom_final))


#Функция, тренирующая KNN классификатор
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose = False):
    X = []
    y = []
    #Проход по всем изображениям в датасете
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))

    #Тренировка KNN классификатора
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Сохранение обученной модели на диск (чтобы не переобучать)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

#Функция, возвращающая предсказание имени человека
def predict(frame, face_locations, knn_clf=None, model_path=None, distance_threshold=0.5): #функция возвращающая предсказание имение пользователя
    if len(face_locations) == 0:  #Если на изображении вообще не найдено лиц, то return
        return []
    faces_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations) #Получаем закодированные изображения
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    """
    Используя, KNN классификатор получаем предсказание (массив с минимальным "расстоянием" до искомого изображения
    """
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]



if __name__ == "__main__":
    last_time = time.time()
    time.sleep(1)
    classifier = train("./dataset", model_save_path="./config/trained_knn_model.clf", n_neighbors=2)
    face_locations = []
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        fast_frame = cv2.resize(frame,(0, 0), fx=0.2, fy=0.2) #Уменьшаю изображение, чтобы ускорить обработку face_encodings
        rgb_frame = fast_frame[:, :, ::-1] #Преобразую в RGB
        was_upd = False
        face_locations = face_recognition.face_locations(rgb_frame)
        if time.time() - last_time > 0.1: #Обновляю 10 раз в секунду prediction
            predictions = predict(rgb_frame, face_locations, knn_clf=classifier, model_path="./config/trained_knn_model.clf")
            names = [item[0] for item in predictions]
            last_time = time.time()
            was_upd = True
        for name, (top, right, bottom, left) in zip(names, face_locations):
            maxim = -1000000000000000000
            top *= 5
            right *= 5
            bottom *= 5
            left *= 5
            """
            if not was_upd:
                name_found = None
                for name, (top_prev, right_prev, bottom_prev, left_prev) in predictions:
                    current_square = square_intersection(left=left, right=right, bottom=bottom, top=top, left2=left_prev, right2=right_prev, bottom2=bottom_prev, top2=top_prev)
                    print(current_square)
                    if current_square > maxim:
                        maxim = current_square
                        name_found = name
                print("name found == ", name_found)
            """
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            was_upd = False
        cv2.imshow('Video', frame) #Вывод во фрейм
        if cv2.waitKey(10) == 27:  # Клавиша Esc
            break
    video_capture.release()
    cv2.destroyAllWindows()
