# -*- coding: utf-8 -*-
import dlib
import cv2
import sqlite3
from contextlib import closing
import numpy as np
import os
import csv

dbname = "db_file"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
name_path = "file_path"
name_list = [name for name in os.listdir(name_path) if not name.startswith(".")]
sql_data = {}


def main(input_data, threshold):
    result_dict = {}
    for names, datas in sql_data.items():
        result_dict[str(names)] = np.linalg.norm(input_data-np.array(datas))
    min_result = min(result_dict.items(), key=lambda x: x[1])
    if min_result[1]/6 <= float(threshold):
        return min_result[0]
    else:
        return "not register"


def point(image):
    landmark_dir = []
    rects = detector(image, 1)
    if len(rects) > 0:
        try:
            for rect in rects:
                img = image[rect.top():rect.bottom(), rect.left():rect.right()]
                img = cv2.resize(img, (300, 300))
        except cv2.error:
            top = rect.top()
            bottom = rect.bottom()
            left = rect.left()
            right = rect.right()
            if top < 0:
                top = 0
            if bottom > image.shape[0]:
                bottom = image.shape[0]
            if left < 0:
                left = 0
            if right > image.shape[1]:
                right = image.shape[1]
            img = image[top:bottom, left:right]
            img = cv2.resize(img, (300, 300))
        resize_rects = detector(img, 1)
        for rect in resize_rects:
            landmarks = np.array(
                [[p.x, p.y] for p in predictor(img, rect).parts()]
                )
            if len(landmarks) == 68:
                mouse_top_x = landmarks[51][0]
                mouse_top_y = landmarks[51][1]
                mouse_bottom_x = landmarks[57][0]
                mouse_bottom_y = landmarks[57][1]
                mouse_right_x = landmarks[55][0]
                mouse_right_y = landmarks[55][1]
                mouse_left_x = landmarks[48][0]
                mouse_left_y = landmarks[48][1]
                nose_top_x = landmarks[27][0]
                nose_top_y = landmarks[27][1]
                nose_bottom_x = landmarks[33][0]
                nose_bottom_y = landmarks[33][1]
                nose_right_x = landmarks[35][0]
                nose_right_y = landmarks[35][1]
                nose_left_x = landmarks[31][0]
                nose_left_y = landmarks[31][1]
                right_eye_outside_x = landmarks[45][0]
                right_eye_outside_y = landmarks[45][1]
                right_eye_inside_x = landmarks[42][0]
                right_eye_inside_y = landmarks[42][1]
                left_eye_outside_x = landmarks[36][0]
                left_eye_outside_y = landmarks[36][1]
                left_eye_inside_x = landmarks[39][0]
                left_eye_inside_y = landmarks[39][1]
                right_eyebrow_outside_x = landmarks[26][0]
                right_eyebrow_outside_y = landmarks[26][1]
                right_eyebrow_inside_x = landmarks[22][0]
                right_eyebrow_inside_y = landmarks[22][1]
                left_eyebrow_outside_x = landmarks[17][0]
                left_eyebrow_outside_y = landmarks[17][1]
                left_eyebrow_inside_x = landmarks[21][0]
                left_eyebrow_inside_y = landmarks[21][1]

                landmark_dir.append([[mouse_top_x, mouse_top_y], [mouse_bottom_x, mouse_bottom_y], [mouse_right_x, mouse_right_y],\
                [mouse_left_x, mouse_left_y], [nose_top_x, nose_top_y], [nose_bottom_x, nose_bottom_y],\
                [nose_right_x, nose_right_y], [nose_left_x, nose_left_y], [right_eye_outside_x, right_eye_outside_y],
                [right_eye_inside_x, right_eye_inside_y], [left_eye_outside_x, left_eye_outside_y], [left_eye_inside_x, left_eye_inside_y],
                [right_eyebrow_outside_x, right_eyebrow_outside_y], [right_eyebrow_inside_x, right_eyebrow_inside_y], [left_eyebrow_outside_x, left_eyebrow_outside_y],\
                [left_eyebrow_inside_x, left_eyebrow_inside_y]])

    return landmark_dir, rects



def sql_select():
    for name in name_list:
        select_sql = '''select
                        mouse_top_x ,
                        mouse_top_y ,
                        mouse_bottom_x ,
                        mouse_bottom_y ,
                        mouse_right_x ,
                        mouse_right_y ,
                        mouse_left_x ,
                        mouse_left_y ,
                        nose_top_x ,
                        nose_top_y ,
                        nose_bottom_x ,
                        nose_bottom_y ,
                        nose_right_x ,
                        nose_right_y ,
                        nose_left_x,
                        nose_left_y,
                        right_eye_outside_x ,
                        right_eye_outside_y ,
                        right_eye_inside_x ,
                        right_eye_inside_y ,
                        left_eye_outside_x ,
                        left_eye_outside_y ,
                        left_eye_inside_x ,
                        left_eye_inside_y ,
                        right_eyebrow_outside_x,
                        right_eyebrow_outside_y,
                        right_eyebrow_inside_x,
                        right_eyebrow_inside_y,
                        left_eyebrow_outside_x,
                        left_eyebrow_outside_y,
                        left_eyebrow_inside_x,
                        left_eyebrow_inside_y from {}'''.format(name)
        with closing(sqlite3.connect(dbname)) as conn:
            c = conn.cursor()
            for row in c.execute(select_sql):
                sql_data[str(name)] = np.array([[row[0], row[1]], [row[2], row[3]], [row[4], row[5]], [row[6], row[7]],\
                                        [row[8], row[9]], [row[10], row[11]], [row[12], row[13]], [row[14], row[15]],\
                                        [row[16], row[17]], [row[18], row[19]], [row[20], row[21]], [row[22], row[23]],\
                                        [row[24], row[25]], [row[26], row[27]], [row[28], row[29]], [row[30], row[31]]])



if __name__ == "__main__":
    ESC_KEY = 27   # Escキー
    INTERVAL = 33   # 待ち時間
    FRAME_RATE = 30  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0
    threshold = 58

    # 分類器の指定
    detector = dlib.get_frontal_face_detector()
    PREDICTOR_PATH = "/Users/etokazuki/pyfolder/dlib-18.18/examples/build/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    sql_select()
    # 変換処理ループ
    while cap.isOpened() is True:
        result_name = []

        ret, frame = cap.read()

        height, width = frame.shape[:2]
        temp_frame = cv2.resize(frame, (int(width), int(height)))
        landmark_dir, rects = point(temp_frame)

        for j in landmark_dir:
            result_name.append(main(j, threshold))

        for draw_name, rect in zip(result_name, rects):
            cv2.rectangle(temp_frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 0, 255), 2)
            cv2.rectangle(temp_frame, (rect.left(), rect.bottom() - 35), (rect.right(), rect.bottom()), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(temp_frame, draw_name, (rect.left() + 6, rect.bottom() - 6), font, 1.0, (255, 255, 255), 1)
        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, temp_frame)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        # ret, frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
