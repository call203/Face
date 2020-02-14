# -*- coding: utf-8 -*-
#ROI : 사용자가 보고 싶은 곳만 볼수 있게
import cv2
import sys
import dlib
import numpy as np

scaler = 0.3

detector = dlib.get_frontal_face_detector()
# 머신러닝으로 학습된 모델
predictor = dlib.shape_predictor("C:/Users/call2/PycharmProjects/video/shape_predictor_68_face_landmarks.dat")
#비디오 로드
cap = cv2.VideoCapture('C:/Users/call2/data/video.mp4')

while True:
  # read frame buffer from video
  ret, img = cap.read()
  if not ret:
    break

  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0]*scaler)))
  #원본 이미지 복사
  ori = img.copy()

  #detect faces 월굴하나(0번 인덱스만 저장)
  faces = detector(img)
  face = faces[0]

  #얼굴 특징 추출
  # 이미지와 얼굴 영역 들어감
  # 연산을 쉽게 하기위해 numpy array바꾸어서 shap_2d에 저장
  dlib_shape = predictor(img, face)
  shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

  #compute center of face // 좌상단, 우하단을 이용해 얼굴의 사이즈 구하기
  top_left = np.min(shape_2d, axis=0)
  bottom_right = np.max(shape_2d, axis=0)
  #얼굴 중심 구하기
  center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int) #정수형 변환


  #visualize
  img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255),
                      thickness=2, lineType=cv2.LINE_AA)
  #얼굴 특징점의 갯수는 68개이다.
  #64개의 점은 opencv에 circle이라는 method로 그려준다.
  for s in shape_2d:
    cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

  cv2.circle(img, center=tuple(top_left), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
  cv2.circle(img, center=tuple(bottom_right), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

  cv2.circle(img, center=tuple((center_x, center_y)), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

  cv2.imshow('img', img)
  cv2.waitKey(1)
