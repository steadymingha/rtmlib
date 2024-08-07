import time

import cv2

from rtmlib import Body, draw_skeleton
from pose_tools import Crossfit, Sort

import numpy as np

device = 'cpu'
backend = 'onnxruntime'  # opencv, onnxruntime, openvino
video_dir = '../gmj/GMJ_24.1_burpee_21.mp4'
id = 0


# SORT 객체 생성
sort = Sort()

cap = cv2.VideoCapture(video_dir)

openpose_skeleton = False  # True for ovideo_dirpenpose-style, False for mmpose-style

model = Body(
    # pose='rtmo',
    to_openpose=openpose_skeleton,
    mode='performance',  # balanced, performance, lightweight
    backend=backend,
    device=device)

frame_idx = 0

judge = Crossfit()
while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break
    s = time.time()

    bboxes = model.det_model(frame)
    keypoints, scores = model.pose_model(frame, bboxes=bboxes)

    # # choose object id and display the output
    # bbox = sort.check_athlete(bboxes, id)
    # cv2.putText(frame, f'ID: {id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # SORT - object tracking
    tracked_objects = sort.update(bboxes)

    for obj in tracked_objects:
        bbox = obj[:4]
        object_id = int(obj[4])
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {object_id}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0)  , 2)




    img_show = frame.copy()

    # if you want to use black background instead of original image,
    # img_show = np.zeros(img_show.shape, dtype=np.uint8)

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.3,
                             line_width=2)

    img_show = cv2.resize(img_show, (960, 640))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)
