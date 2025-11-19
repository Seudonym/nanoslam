#! /usr/bin/env python3

import pygame
import time
import cv2
from cv2.typing import MatLike
import numpy as np
from visual_odometry import VisualOdometry


def paint(display: pygame.Surface, img: MatLike):
    transformed_img = cv2.transpose(img, None)
    transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
    surface = pygame.surfarray.make_surface(transformed_img)
    display.blit(surface, (0, 0))


def process_frame(vo: VisualOdometry, img: MatLike):
    img = cv2.resize(img, (W, H))
    matches, pose = vo.extract(img)

    for pt1, pt2 in matches:
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

    F = 610
    K = np.array(
        [
            [F, 0, W // 2],
            [0, F, H // 2],
            [0, 0, 1],
        ]
    )
    vo = VisualOdometry(K)

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

            frame = process_frame(vo, frame)
            paint(display, frame)
            pygame.display.flip()
            clock.tick(10)

    pygame.quit()
