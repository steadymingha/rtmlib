import time

import cv2
import numpy as np
from rtmlib import Body, draw_skeleton
from pose_tools import Crossfit, Sort, normalize_keypoints



video_dir = '../gmj/GMJ_24.1_burpee_21.mp4'
cap = cv2.VideoCapture(video_dir)
id = 3


sort = Sort()
judge = Crossfit()
model = Body(device='cuda')

while cap.isOpened():
    ret, frame = cap.read()

    bboxes = model.det_model(frame)

    bbox_id, bbox = sort.choose_athlete(bboxes, id)

    bbox = bbox[None, ...]
    keypoints, scores = model.pose_model(frame, bboxes=bbox)

    bbox = bbox.flatten()
    kpt = normalize_keypoints(bbox, keypoints[id], scores[id])
    # kpt = keypoints[bbox_idx]
    # kpt = kpt[None, ...]
    img_show = frame.copy()






    # img_show = draw_skeleton(img_show,
    #                          keypoints,
    #                          scores,
    #                          kpt_thr=0.3,
    #                          line_width=2)

    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
