# -*- coding: utf-8 -*-

import dlib
import cv2
import sqite3
from contextlib import closing
import numpy as np
import path
from os import argv

dbname = "register_face.db"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("file_path")
images_path = argv[1]
images_list = path.listdir(images_path)


def register(name, image):
    with closing(sqite3.connect(dbname)) as conn:
        c = conn.cursor()

        create_table = '''create table if not exists {} (
                          id integer primary key autoincrement,
                          eye_surrounding real,
                          eye_distance real,
                          eye2eyebrw real,
                          mouse_surrounding real,
                          mouse_distance real,
                          mouse2nose real,
                          base_distance real)
                          '''.format(name)
        c.execute(create_table)

        sql = '''insert into {}
                 (eye_surrounding,
                  eye_distance,
                  eye_eyebrow,
                  mouse_surrounding,
                  mouse_distance,
                  base_distance)
                  values (?,?,?,?,?,?)'''.format(name)
        eye_surrounding, eye_distance, eye_eyebrow, mouse_surrounding, mouse_distance, base_distance = point2point(image)
        user = (eye_surrounding, eye_distance, eye_eyebrow, mouse_surrounding, mouse_distance, base_distance)
        c.execute(sql, user)
        conn.commit()


def point2point(image):
    right_eye_sum = 0.0
    left_eye_sum = 0.0
    left_eye_braw = 0.0
    right_eye_braw = 0.0
    mouse_surrounding = 0.0
    mouse_distance = 0.0
    eye_eyebrow = 0.0
    eye_distance = 0.0
    base_distance = 0.0
    # 点と点の距離を格納しておくリスト
    image = cv2.imread(image)
    rects = detector(image, 1)
    for rect in rects:
        landmarks = np.array(
            [[p.x, p.y] for p in predictor(image, rect).parts()]
            )

    if len(rects) > 0 and len(landmarks) == 68:
        for i in range(36, 41):
            left_eye_sum = left_eye_sum + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+1][0], landmarks[i+1][1])
        left_eye_sum = left_eye_sum + cul_distance(landmarks[41][0], landmarks[41][1], landmarks[36][0], landmarks[36][1])

        for i in range(42, 47):
            right_eye_sum = right_eye_sum + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+1][0], landmarks[i+1][1])
        right_eye_sum = right_eye_sum + cul_distance(landmarks[47][0], landmarks[47][1], landmarks[42][0], landmarks[42][1])

        eye_surrounding = right_eye_sum + left_eye_sum

        for i in range(17, 19):
            left_eye_braw = left_eye_braw + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+19][0], landmarks[i+19][1])

        for i in range(20, 22):
            left_eye_braw = left_eye_braw + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+18][0], landmarks[i+18][1])

        for i in range(25, 27):
            right_eye_braw = right_eye_braw + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+19][0], landmarks[i+19][1])

        for i in range(22, 24):
            right_eye_braw = right_eye_braw + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+20][0], landmarks[i+20][1])

        eye_eyebrow = right_eye_braw + left_eye_braw

        mouse_distance = cul_distance(landmarks[48][0], landmarks[48][1], landmarks[54][0], landmarks[54][1])

        for i in range(48, 59):
            mouse_surrounding = mouse_surrounding + cul_distance(landmarks[i][0], landmarks[i][1], landmarks[i+1][0], landmarks[i+1][1])
        mouse_surrounding = mouse_surrounding + cul_distance(landmarks[48][0], landmarks[48][1], landmarks[59][0], landmarks[59][1])

        eye_distance = cul_distance(landmarks[39][0], landmarks[39][1], landmarks[42][0], landmarks[42][1])

        base_distance = cul_distance(landmarks[58][0], landmarks[58][0], landmarks[9][0], landmarks[9][1])
    else:
        pass
    return eye_surrounding, eye_distance, eye_eyebrow, mouse_surrounding, mouse_distance, base_distance


def cul_distance(x1, y1, x2, y2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    y1 = np.array(y1)
    y2 = np.array(y2)
    dis = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dis


if __name__ == "__main__":
    name = input("名前を入力してください")
    for file in images_list:
        root, exi = path.splitext(images_path+file)
        if exi == ".jpg" or exi == ".png":
            register(images_path+file, name)
        else:
            pass
