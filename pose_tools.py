import numpy as np

jnm = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}
def angle_between_joints(kpt, joint1, joint_mid, joint2):#wx, wy, ex, ey, sx, sy):

    wx, wy = kpt[jnm[joint1]][0], kpt[jnm[joint1]][1]
    ex, ey = kpt[jnm[joint_mid]][0], kpt[jnm[joint_mid]][1]
    sx, sy = kpt[jnm[joint2]][0], kpt[jnm[joint2]][1]
    # Vector from wrist to elbow
    # Vector from elbow to shoulder
    vec_we = np.array([wx - ex, wy - ey])
    # vec_we = np.array([ex - wx, ey - wy])
    vec_es = np.array([sx - ex, sy - ey])

    # Calculate the dot product
    dot_product = np.dot(vec_we, vec_es)

    # Calculate the magnitudes of the vectors
    mag_we = np.linalg.norm(vec_we)
    mag_es = np.linalg.norm(vec_es)

    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product / (mag_we * mag_es))

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# keypoints output : (17,2)

class Crossfit:
    def __init__(self):
        self.prev_joint = 0
        self.phase = 0
    def db_r_snatch(self, kpt, rep_count, threshold=165):
        ## start / end condition
        if kpt[jnm['right_wrist']][1] > kpt[jnm['right_knee']][1] \
                and kpt[jnm['right_shoulder']][1] > kpt[jnm['left_shoulder']][1]:  ## wrist is lower than knee , right shoulder is lower tan l shoulder start
            self.phase = 0

        elif kpt[jnm['right_wrist']][1] < kpt[jnm['right_eye']][1] \
                and kpt[jnm['right_wrist']][1] < self.prev_joint and self.phase == 0:  ## end (ascending) : wrist is higher than eye, wrist is higher last location

            ang = angle_between_joints(kpt, 'right_wrist', 'right_elbow', 'right_shoulder')

            ## rule
            if ang > threshold and self.phase is not None:
                rep_count += 1
                self.phase = None

        self.prev_joint = kpt[jnm['right_wrist']][1]

        return rep_count

    def db_l_snatch(self, kpt, rep_count, threshold=165):
        ## start / end condition
        if kpt[jnm['left_wrist']][1] > kpt[jnm['left_knee']][1] \
                and kpt[jnm['left_shoulder']][1] > kpt[jnm['right_shoulder']][1]:  ## wrist is lower than knee , left shoulder is lower tan l shoulder start
            self.phase = 0

        elif kpt[jnm['left_wrist']][1] < kpt[jnm['left_eye']][1] \
                and kpt[jnm['left_wrist']][1] < self.prev_joint and self.phase == 0:  ## end (ascending) : wrist is higher than eye, wrist is higher last location

            ang = angle_between_joints(kpt, 'left_wrist', 'left_elbow', 'left_shoulder')

            ## rule
            if ang > threshold and self.phase is not None:
                rep_count += 1
                self.phase = None

        self.prev_joint = kpt[jnm['left_wrist']][1]

        return rep_count

    def burpee_db_jump_over(self, kpt, rep_count):
        if kpt[jnm['right_shouler']][1] > kpt[jnm['right_elbow']][1] \
            and kpt[jnm['right_ankle']][1] < kpt[jnm['left_shoulder']][1]: # lying down
            self.phase = 0


    def __call__(self):
        self.__init__()


import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R *= 10.
        self.kf.P *= 10.
        self.kf.Q *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        else:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))



class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trks, self.iou_threshold)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(detections[d, :][0])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def choose_athlete(self, detections, id):
        tracked_objects = self.update(detections)

        for i, obj in enumerate(tracked_objects):
            bbox = obj[:4]
            ob_id = int(obj[4])
            if id == ob_id:
                return i, bbox

        raise





def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def normalize_keypoints(bbox, keypoints, scores):
    keypoints[:, 0] = keypoints[:, 0] - bbox[0]
    keypoints[:, 1] = keypoints[:, 1] - bbox[1]
    keypoints[:, 0] = keypoints[:, 0].clip(0)
    keypoints[:, 1] = keypoints[:, 1].clip(0)

    keypoints[:, 0] = (keypoints[:, 0] - keypoints[:, 0].min()) / (keypoints[:, 0].max() - keypoints[:, 0].min() + 1e-6)
    keypoints[:, 1] = (keypoints[:, 1] - keypoints[:, 1].min()) / (keypoints[:, 1].max() - keypoints[:, 1].min() + 1e-6)

    keypoints = keypoints.reshape(-1)
    return keypoints