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
            # was using for estimating E manually
            # ret[:, 0, :] = (
            #     self.Kinv
            #     @ np.concat([ret[:, 0, :], np.ones((ret.shape[0], 1))], axis=-1).T
            # ).T[:, 0:2]
            #
            # ret[:, 1, :] = (
            #     self.Kinv
            #     @ np.concat([ret[:, 1, :], np.ones((ret.shape[0], 1))], axis=-1).T
            # ).T[:, 0:2]

            # estimate the essential matrix using RANSAC
            E, mask = cv2.findEssentialMat(ret[:, 0], ret[:, 1], self.K, cv2.RANSAC)
            _, R, t, _ = cv2.recoverPose(E, ret[:, 0], ret[:, 1], self.K)

            # filter ransac inliers
            ret = ret[mask.ravel() == 1]
            pose = np.concatenate([R, t], axis=1)
            print(t.ravel())

        self.last = {"kps": kps, "des": des}
        return ret, pose
