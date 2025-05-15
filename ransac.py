"""Ransac loop."""

import cv2 as cv
import numpy as np
import random


def ransac_loop(points1, points2, iterations=1000, threshold=20):
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
        # sample 2 random pair of points
        idx = random.sample(range(len(points1)), 2)
        pair_1 = np.array([points1[idx[0]], points2[idx[0]]])
        pair_2 = np.array([points1[idx[1]], points2[idx[1]]])

        # find rotation based on 2 pairs of points
        # TODO: Find a way to find matrix
        M = ...

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
