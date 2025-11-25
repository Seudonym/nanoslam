#! /usr/bin/env python3

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import cv2
from cv2.typing import MatLike
import numpy as np
import matplotlib.pyplot as plt
import pygame

from frame import Frame, match_frames, extract


def paint(display: pygame.Surface, img: MatLike):
    transformed_img = cv2.transpose(img, None)
    transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
    surface = pygame.surfarray.make_surface(transformed_img)
    display.blit(surface, (0, 0))


frames = []


def process_frame(img: MatLike):
    global frames
    img = cv2.resize(img, (W, H))
    frame = Frame(img, K)
    frames.append(frame)

    if len(frames) < 2:
        return img

    matches, Rt = match_frames(frames[-2], frames[-1])
    print(Rt[:3, 3].ravel())

    for match in matches:
        u1, v1 = map(lambda x: int(round(x)), match[0][0:2])
        u2, v2 = map(lambda x: int(round(x)), match[1][0:2])

        img = cv2.circle(
            img, (u1, v1), color=(0, 0, 255), radius=1
        )  # last frame: green
        img = cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))  # match: blue
        img = cv2.circle(
            img, (u2, v2), color=(0, 255, 0), radius=2
        )  # current frame: red

    return img


if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()
    video_file = input("Enter path to video file: ")
    if video_file == "":
        video_file = "videos/test1.mp4"

    cap = cv2.VideoCapture(video_file)
    ret, img = cap.read()
    H, W, _ = img.shape
    cap.release()

    F = 610
    K = np.array(
        [
            [F, 0, W // 2],
            [0, F, H // 2],
            [0, 0, 1],
        ]
    )

    cap = cv2.VideoCapture(video_file)
    display = pygame.display.set_mode((W, H))
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            img = process_frame(img)
            paint(display, img)
            pygame.display.flip()
            clock.tick(10)

    pygame.quit()
