#! /usr/bin/env python3

import pygame
import time
import cv2
from cv2.typing import MatLike
import numpy as np

W, H = 1242, 374


def paint(display: pygame.Surface, img: MatLike):
    transformed_img = cv2.transpose(img, None)
    transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
    surface = pygame.surfarray.make_surface(transformed_img)
    display.blit(surface, (0, 0))


class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB.create(nfeatures=5000)
        self.bf = cv2.BFMatcher.create(normType=cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img: MatLike):
        # feature detection and extraction
        kps, des = self.orb.detectAndCompute(img, None)

        # feature matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last["des"], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    ret.append((kps[m.queryIdx], self.last["kps"][m.trainIdx]))

        self.last = {"kps": kps, "des": des}
        print(f"Matches {len(ret)}")

        return ret


fe = FeatureExtractor()


def process_frame(img: MatLike):
    img = cv2.resize(img, (W, H))
    matches = fe.extract(img)

    for kp1, kp2 in matches:
        u1, v1 = map(lambda x: int(round(x)), kp1.pt)
        u2, v2 = map(lambda x: int(round(x)), kp2.pt)

        img = cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        img = cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    # img = cv2.drawKeypoints(
    #     img,
    #     kps,
    #     outImage=0,
    #     color=(0, 255, 0),
    #     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    # )

    return img


if __name__ == "__main__":
    pygame.init()
    display = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture("0000.mp4")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame)
            paint(display, frame)
            pygame.display.flip()
            clock.tick(10)

    pygame.quit()
