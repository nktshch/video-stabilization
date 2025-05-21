"""RANSAC loop."""

import cv2 as cv
import numpy as np
import random


def find_transform(pair_1, pair_2):
    centers = (pair_1 + pair_2) / 2
    differences = (pair_2 - pair_1) / 2
    differences = differences.dot(np.array([[0, 1], [-1, 0]]))
    pair_3 = centers + differences
    pair_4 = centers - differences
    src = np.array([pair_1[0], pair_2[0], pair_3[0], pair_4[0]], np.float32)
    dst = np.array([pair_1[1], pair_2[1], pair_3[1], pair_4[1]], np.float32)
    M = cv.getPerspectiveTransform(src=dst, dst=src)
    return M


def ransac_loop(points1, points2, iterations=1000, threshold=20, lib_func=False):
    """Performs RANSAC loop.

    1) Sample 2 pair of points. Each pair is [point on an old frame, point on a new frame].
    2) Use the pairs to find transform matrix. 3D rotation can be uniquely determined by 3 parameters (e.g. Euler angles),
    hence 2 pairs (each pair provides 2 params - coordinates in 2D space).
    3) Find outliers. Check if found transform suits other pairs of points - apply it to old points
    and measure distance to new ones (search for RANSAC explanation for details).
    4) Loop is ended after a certain amount of iterations.
    Returns matrix.
    """
    best_inliers = []
    best_model = None

    # RANSAC loop
    for _ in range(iterations):
        if lib_func is True:
            # sample 4 random pairs of points
            idx = random.sample(range(len(points1)), 4)
            src = np.array([points1[i] for i in idx], np.float32)
            dst = np.array([points2[i] for i in idx], np.float32)

            # find rotation using cv function
            M = cv.getPerspectiveTransform(src=dst, dst=src)
        else:
            # sample 2 random pairs of points
            idx = random.sample(range(len(points1)), 2)
            pair_1 = np.array([points1[idx[0]], points2[idx[0]]])
            pair_2 = np.array([points1[idx[1]], points2[idx[1]]])

            # find rotation based on 2 pairs of points
            # TODO: Find a way to find matrix
            M = find_transform(pair_1, pair_2)

        # find inliers
        inliers = []
        for i in range(len(points1)):
            # apply transformation and compute distance
            transformed = np.dot(M, np.append(points2[i], 1))
            distance = np.linalg.norm(transformed - np.append(points1[i], 1))

            # if the distance is below the threshold, it’s an inlier
            if distance < threshold:
                inliers.append(i)

        # track the best model (with most inliers)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = M

        # consider adding a condition for early stop
        # consider changing the way of selecting the best model (not the most inliers, smth else)

    # print("Best matrix:", best_model)
    # print("Number of inliers:", len(best_inliers))

    return best_model


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    p1 = np.array([[360, 560], [360, 600]])
    p2 = np.array([[360, 720], [320, 750]])

    fig, ax = plt.subplots(2, 1, figsize=(16, 18))

    ax[0].imshow(np.zeros((720, 1280)), cmap="cividis")
    ax[1].imshow(np.zeros((720, 1280)), cmap="cividis")
    ax[0].scatter([p1[0][1], p2[0][1]], [p1[0][0], p2[0][0]], c="white", marker=".")
    ax[1].scatter([p1[1][1], p2[1][1]], [p1[1][0], p2[1][0]], c="white", marker=".")
    c = (p1 + p2) / 2
    ax[0].scatter([c[0][1]], [c[0][0]], c="white", marker="x")
    ax[1].scatter([c[1][1]], [c[1][0]], c="white", marker="x")
    d = (p2 - p1) / 2
    d = d.dot(np.array([[0, 1], [-1, 0]]))
    p3 = c + d
    p4 = c - d
    ax[0].scatter([p3[0][1], p4[0][1]], [p3[0][0], p4[0][0]], c="red", marker=".")
    ax[1].scatter([p3[1][1], p4[1][1]], [p3[1][0], p4[1][0]], c="red", marker=".")

    fig.savefig("points.png")
    # some_M = find_transform(p1, p2)

    img = cv.imread("test1.png")
    height, width = img.shape[:2]
    test_M = np.array([[0.7, -0.7, 100],
                       [0.7, 0.7, 700],
                       [0, 0, 1]], np.float32)
    warped_img = cv.warpPerspective(img, test_M, (width, height))
    cv.imwrite("warped_img.png", warped_img)
