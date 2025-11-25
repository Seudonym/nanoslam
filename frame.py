import cv2
from cv2.typing import MatLike
import numpy as np


def extract(img: MatLike):
    orb = cv2.ORB.create(nfeatures=5000)
    kps, des = orb.detectAndCompute(img, None)
    return kps, des


class Map:
    def __init__(self):
        self.frames: list[Frame] = []
        self.points: list[Point] = []


class Frame:
    def __init__(self, map: Map, img: MatLike, K: np.ndarray):
        self.kps, self.des = extract(img)
        self.pose = np.eye(4)
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.id = len(map.frames)
        map.frames.append(self)


class Point:
    def __init__(self, map: Map, location):
        self.location = location
        self.frames = []
        self.idxs = []
        self.id = len(map.points)
        map.points.append(self)

    def add_observation(self, frame: Frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def match_frames(frame_q: Frame, frame_t: Frame):
    K = frame_t.K
    bf = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)

    ret = []
    idxs1 = []
    idxs2 = []
    # Lowe's ratio test
    matches = bf.knnMatch(frame_q.des, frame_t.des, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            idxs1.append(m.queryIdx)
            idxs2.append(m.trainIdx)

            pt1 = frame_q.kps[m.queryIdx].pt
            pt2 = frame_t.kps[m.trainIdx].pt
            ret.append((pt1, pt2))

    assert len(ret) >= 8

    idxs1 = np.array(idxs1)
    idxs2 = np.array(idxs2)
    ret = np.array(ret)

    Rt = None

    # estimate the essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(ret[:, 1], ret[:, 0], K, cv2.RANSAC)
    # ret = ret[mask.ravel() == 1]
    filter = mask.ravel() == 1
    ret = ret[filter]
    idxs1 = idxs1[filter]
    idxs2 = idxs2[filter]

    _, R, t, _ = cv2.recoverPose(E, ret[:, 1], ret[:, 0], K)
    Rt = np.concat([R, t], axis=1)
    Rt = np.vstack([Rt, [0, 0, 0, 1]])

    return idxs1, idxs2, Rt
