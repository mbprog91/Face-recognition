Описание программы:

Программа написана на языке Python, с использованием face-recognition-api, dlib, KNN классификатора из scikit-learn и opencv (cv2). Распознает лица, находящиеся в кадре вебкамеры и ищет совпадение в датасете; загруженном пользователем. Если совпадение есть, то выводит имя распознанного человека, если нет, то обозначает его как "unknown".

Проблемы, возникшие при написании кода:
1) Установка необходимых библиотек (да-да, это тоже может быть проблемой).

Решение: кровью, потом и гуглом

2) Очень низкое качество распознавания лица.
Изначально, KNN классификатор от scikit-learn подключен не был, а использовалась встроенная функция сравнения лиц от face-recognition-api. К сожалению, работала она очень плохо и в подавляющем большинстве случаев правильно находила только те случаи, изображения идентичные (или очень похожие) которым уже подгружены в датасет (а понятно, что загрузить изображения лица человека во всех возможных ракурсах, освещениях, позициях и фонах - невозможно)

Решение: вместо встроенной функции сравнения лиц подключен KNN классификатор из scikit-learn, возвращающий информацию о найденном совпадении, или "unknown", если совпадение не найдено

3) Ложные срабатывания."unknown" возвращался крайне редко, а людям, которых нет в датасете присваивалось неправильное значение из набора тех, кто в нём есть (вместо "unknown").

Решение: изменение параметра distance_threshold, отвечающего за погрешность, позволяющую соотнести пользователя с изображением в датасете (чем distance_threshold больше, тем больше вероятность ложных срабатываний) с 0.6 до 0.5. Между прочим, пришлось подбирать!

4) Медленная работа face-encoding, создающая задержки вывода изображения во фрейм

Решение:
Обновление параметров face-encoding раз в 0.1 секунду, при этом обводка лиц по списку face_locations параллельно с выводом изображения во фрейм (чтобы лица были обведены постоянно). Обновление раз в 0.1 секунду не влияет на качество и не видно глазу, но при этом заметно ускоряет программу.



Нерешенные проблемы и потенциальные пути их решения:


1) Наименование пользователей в датасете может осуществляться только на латинице. Если использовать кириллицу, то вместо подписи, отображающей имя пользователя, будут отображаться знаки вопроса

Потенциальное решение:
Найти и подгрузить шрифты opencv для кириллицы. Для китайского языка они существуют, значит для русского - тоже (Л-Логика)

2) Редкие(очень) ошибки при распознавании, образующиеся из-за очень сильно различающегося количества изображений в датасете по пользователям.

Потенциальное решение:
Нормирование количества изображений по пользователям до одинакового количества
