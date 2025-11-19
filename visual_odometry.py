import cv2
from cv2.typing import MatLike
import numpy as np


class VisualOdometry:
    def __init__(self, K: np.ndarray):
        self.orb = cv2.ORB.create(nfeatures=5000)
        self.bf = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.last = None

        self.cur_t = np.zeros((3, 1))
        self.cur_R = np.eye(3)
        self.path = []

    def extract(self, img: MatLike) -> tuple[np.ndarray, np.ndarray | None]:
        # feature detection and extraction using ORB
        kps, des = self.orb.detectAndCompute(img, None)

        # feature matching with lowe's ratio test
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last["des"], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    pt1 = self.last["kps"][m.trainIdx].pt
                    pt2 = kps[m.queryIdx].pt
                    ret.append((pt1, pt2))
        ret = np.array(ret)

        pose = None
        if len(ret) > 0:
            # estimate the essential matrix using RANSAC
            E, mask = cv2.findEssentialMat(ret[:, 0], ret[:, 1], self.K, cv2.RANSAC)
            ret = ret[mask.ravel() == 1]
            _, R, t, _ = cv2.recoverPose(E, ret[:, 0], ret[:, 1], self.K)

            self.cur_t = self.cur_t + self.cur_R @ t
            self.cur_R = R @ self.cur_R
            self.path.append(self.cur_t.ravel().tolist())

            # filter ransac inliers
            pose = np.concatenate([R, t], axis=1)
            print(t.ravel())

        self.last = {"kps": kps, "des": des}
        return ret, pose
