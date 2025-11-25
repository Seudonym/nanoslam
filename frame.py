import cv2
from cv2.typing import MatLike
import numpy as np


def extract(img: MatLike):
    orb = cv2.ORB.create(nfeatures=5000)
    kps, des = orb.detectAndCompute(img, None)
    return kps, des


class Frame:
    def __init__(self, img: MatLike, K: np.ndarray):
        self.kps, self.des = extract(img)
        self.K = K
        self.Kinv = np.linalg.inv(K)


def match_frames(frame1: Frame, frame2: Frame):
    K = frame1.K
    bf = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)

    ret = []
    # Lowe's ratio test
    matches = bf.knnMatch(frame1.des, frame2.des, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pt1 = frame1.kps[m.queryIdx].pt
            pt2 = frame2.kps[m.trainIdx].pt
            ret.append((pt1, pt2))

    assert len(ret) >= 8

    ret = np.array(ret)
    Rt = None

    # estimate the essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(ret[:, 0], ret[:, 1], K, cv2.RANSAC)
    ret = ret[mask.ravel() == 1]

    _, R, t, _ = cv2.recoverPose(E, ret[:, 0], ret[:, 1], K)
    Rt = np.concat([R, t], axis=1)

    return ret, Rt
