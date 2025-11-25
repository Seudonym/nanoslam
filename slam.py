#! /usr/bin/env python3

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import cv2
from cv2.typing import MatLike
import numpy as np
import pygame

from frame import Frame, Point, Map, match_frames, extract

map3d = Map()


def paint(display: pygame.Surface, img: MatLike):
    transformed_img = cv2.transpose(img, None)
    transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)
    surface = pygame.surfarray.make_surface(transformed_img)
    display.blit(surface, (0, 0))


def process_frame(img: MatLike):
    img = cv2.resize(img, (W, H))
    frame = Frame(map3d, img, K)

    frames = map3d.frames
    if len(frames) < 2:
        return img

    idxs1, idxs2, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose = Rt @ frames[-2].pose

    pts1 = np.array([kp.pt for kp in frames[-1].kps])[idxs1]
    pts2 = np.array([kp.pt for kp in frames[-2].kps])[idxs2]

    pts4d = cv2.triangulatePoints(
        frames[-1].pose[:3],
        frames[-2].pose[:3],
        pts1.T,
        pts2.T,
    ).T

    # reject points behind the camera
    filter = pts4d[:, 2] > 0
    pts4d = pts4d[filter]
    pts4d /= pts4d[:, 3:]

    for i, pt in enumerate(pts4d):
        p = Point(map3d, pt)
        p.add_observation(frames[-1], idxs1[i])
        p.add_observation(frames[-2], idxs2[i])

    for pt1, pt2 in zip(pts1, pts2):
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)

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
        video_file = "videos/0000.mp4"

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
            clock.tick(2)

    pygame.quit()
