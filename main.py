"""Main file."""

# not sure if it's really necessary to split such small project into 2 files
import ransac

import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
import math
from pathlib import Path
import os
import glob


np.set_printoptions(precision=3, suppress=True)


def parse_arguments():
    argp = argparse.ArgumentParser()
    argp.add_argument('file', help='Video file')
    argp.add_argument('func', help='Function used to find transform')
    args = argp.parse_args()
    return args


def empty_folder(folder):
    files = glob.glob(f"{folder}/*")
    for f in files:
        os.remove(f)

def mp42img(mp4="1.mp4"):
    """Turns mp4 to list of frames. Also retrieves framerate."""
    cap = cv.VideoCapture(mp4)
    fps = cap.get(cv.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame (end of stream?)")
            break
        frames.append(frame)

    cap.release()
    # cv.destroyAllWindows()
    print(f"Sequence length: {len(frames)}; Framerate: {fps}")
    return frames, fps


def img2mp4(frames, fps, mp4):
    """Turns list of frames to mp4."""
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter(mp4, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    for i, frame in enumerate(frames):
        video_writer.write(frame)

    video_writer.release()


def describe(frame, detector_name):
    """Finds and describes keypoints."""
    if not detector_name:
        raise KeyError("Detector name is not given")
    elif detector_name == "sift":
        detector = cv.SIFT_create()
    elif detector_name == "orb":
        detector = cv.ORB_create(nfeatures=50)
    else:
        raise NotImplementedError("Unknown detector name")

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    frame = cv.drawKeypoints(frame, keypoints, None)
    # cv.imwrite("keypoints.jpg", frame)
    return keypoints, descriptors, frame


def stabilize(frames, descriptor="sift", func="square", lag_behind=1):
    """Uses RANSAC to warp each frame to match previous.

    Initially all frames are warped to match the first. If at a certain frame warp is not possible
    (not enough matches or no good transform found), the last warped frame becomes the reference. The process repeats.
    If warp is still impossible, you are expelled from MIPT.
    """
    bf = cv.BFMatcher(cv.NORM_L2) # consider other matchers

    warped_frames = [frames[0]] # first frame is not warped
    warped_frames_w_kp = [frames[0]]

    old_frame = frames[0] # a.k.a reference frame
    kp1, d1, frame_w_kp1 = describe(old_frame, descriptor)

    # construct inverse camera matrix
    f = 1000 # arbitrary value
    cy, cx = old_frame.shape[:2]
    cx /= 2
    cy /= 2
    C = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    C_inv = np.linalg.inv(C)

    def get_transform(_d1, _d2, _new_frame):
        matches = bf.knnMatch(_d1, _d2, k=2)

        good = []  # used for RANSAC loop
        for m, n in matches:
            if m.distance < 0.6 * n.distance:  # consider changing the threshold (the web says 0.75)
                good.append(m)
        if len(good) < 25:
            return None
        elif len(good) > 100:
            good = sorted(good, key=lambda x: x.distance)[:100]

        good_for_display = [[m] for m in good]

        matches_image = cv.drawMatchesKnn(old_frame, kp1, _new_frame, kp2, good_for_display, None,
                                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(f"matches/matches_{i}.jpg", matches_image)

        points1 = np.array([kp1[m.queryIdx].pt for m in good])  # points on a reference frame
        points2 = np.array([kp2[m.trainIdx].pt for m in good])  # points on a new frame

        M = ransac.ransac_loop(points1, points2, threshold=5, func=func, C_inv=C_inv)

        if func == "svd":
            return C @ M @ C_inv

        return M

    for i, new_frame in enumerate(tqdm(frames[lag_behind:])):
        kp2, d2, frame_w_kp2 = describe(new_frame, descriptor)

        transform = get_transform(d1, d2, new_frame)

        if transform is None:
            # change reference frame and describe
            old_frame = warped_frames[-1]
            kp1, d1, frame_w_kp1 = describe(old_frame, descriptor)

            # find transform again
            transform = get_transform(d1, d2, new_frame)
            if transform is None:
                # TODO: come up with what to do if changing reference to the most recent frame didn't help
                print(f"\nNone at {i} :(")
                return warped_frames

        height, width = old_frame.shape[:2]

        # print("\ntransform")
        # print(transform)

        warped_frame = cv.warpPerspective(new_frame, transform, (width, height))
        warped_frame_w_kp = cv.warpPerspective(frame_w_kp2, transform, (width, height))

        warped_frames.append(warped_frame)
        warped_frames_w_kp.append(warped_frame_w_kp)
        cv.imwrite(f"warped/{i}.jpg", warped_frame_w_kp)

    return warped_frames


def main():
    empty_folder("matches")
    empty_folder("warped")
    args = parse_arguments()
    file = args.file
    func = args.func

    descriptor = "sift"  # probably works best
    sequence, framerate = mp42img(f"{file}")
    new_sequence = stabilize(sequence, descriptor=descriptor, func=func, lag_behind=1)
    new_file = f"{descriptor}-{func}-{file}"
    img2mp4(new_sequence, framerate, new_file)


if __name__ == "__main__":
    main()
