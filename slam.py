#! /usr/bin/env python3

import pygame
import time
import cv2
from cv2.typing import MatLike
import numpy as np


def paint(display: pygame.Surface, img: MatLike):
    transformed_img = cv2.transpose(img, None)
    transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
    surface = pygame.surfarray.make_surface(transformed_img)
    display.blit(surface, (0, 0))


def extractRt(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(E)
    print(d)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return Rt


class FeatureExtractor:
    def __init__(self, K: np.ndarray):
        self.orb = cv2.ORB.create(nfeatures=5000)
        self.bf = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.last = None

    def extract(self, img: MatLike):
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
            E, mask = cv2.findEssentialMat(ret[:, 0], ret[:, 1], K, cv2.RANSAC)
            _, R, t, _ = cv2.recoverPose(E, ret[:, 0], ret[:, 1], self.K)

            # filter ransac inliers
            ret = ret[mask.ravel() == 1]
            pose = np.concatenate([R, t], axis=1)

        self.last = {"kps": kps, "des": des}
        return ret, pose


def process_frame(fe: FeatureExtractor, img: MatLike):
    img = cv2.resize(img, (W, H))
    print(img.shape)
    matches, pose = fe.extract(img)

    for pt1, pt2 in matches:
        # pt1 = fe.K @ np.array([pt1[0], pt1[1], 1.0])
        # pt2 = fe.K @ np.array([pt2[0], pt2[1], 1.0])
        # was using for esimating E manually

        u1, v1 = map(lambda x: int(round(x)), pt1[0:2])
        u2, v2 = map(lambda x: int(round(x)), pt2[0:2])

        img = cv2.circle(
            img, (u1, v1), color=(0, 255, 0), radius=3
        )  # last frame: green
        img = cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))  # match: blue
        img = cv2.circle(
            img, (u2, v2), color=(0, 0, 255), radius=3
        )  # current frame: red

    return img


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    video_file = "0000.mp4"

    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    cap.release()

    F = 280
    K = np.array(
        [
            [F, 0, W // 2],
            [0, F, H // 2],
            [0, 0, 1],
        ]
    )
    fe = FeatureExtractor(K)

    cap = cv2.VideoCapture(video_file)
    display = pygame.display.set_mode((W, H))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(fe, frame)
            paint(display, frame)
            pygame.display.flip()
            clock.tick(1)

    pygame.quit()
