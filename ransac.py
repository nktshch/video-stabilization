"""RANSAC loop."""

import cv2 as cv
import numpy as np
import random


np.set_printoptions(precision=3, suppress=True)


def find_transform(pair_1, pair_2):
    """Finds perspective transform treating each 2 points as square angles."""
    centers = (pair_1 + pair_2) / 2
    differences = (pair_2 - pair_1) / 2
    differences = differences.dot(np.array([[0, 1], [-1, 0]]))

    # find 2 other squares
    pair_3 = centers + differences
    pair_4 = centers - differences

    # find transform using OpenCV
    src = np.array([pair_1[0], pair_2[0], pair_3[0], pair_4[0]], np.float32)
    dst = np.array([pair_1[1], pair_2[1], pair_3[1], pair_4[1]], np.float32)
    M = cv.getPerspectiveTransform(src=dst, dst=src)
    return M


def find_rotation(A, B):
    """Finds rotation using SVD decomposition."""

    centroidA = np.mean(A, axis=0)
    centroidB = np.mean(B, axis=0)

    # A -= centroidA
    # B -= centroidB

    H = np.dot(A.T, B)
    U, S, V = np.linalg.svd(H)

    R = np.dot(V, U.T)
    if np.linalg.det(R) < 0:
        V = np.multiply(V, [1, 1, -1])
        R = np.dot(V, U.T)

    return R


def ransac_loop(points1, points2, iterations=1000, threshold=5, func="square", C_inv=None):
    """Performs RANSAC loop.

    1) Sample 2 pair of points. Each pair is [point on an old frame, point on a new frame].
    2) Use the pairs to find transform matrix. 3D rotation can be uniquely determined by 3 parameters (e.g. Euler angles),
    hence 2 pairs (each pair provides 2 params - coordinates in 2D space).
    3) Find outliers. Check if found transform suits other pairs of points - apply it to old points
    and measure distance to new ones (search for RANSAC explanation for details).
    4) Loop is ended after a certain amount of iterations.
    Returns matrix.
    """

    best_model = None
    best_inliers = []
    best_distances = [1e8]
    best_randoms = None

    # for SVD decomposition, 3D coordinates are used
    spatial_1 = []
    spatial_2 = []
    for point1, point2 in zip(points1, points2):
        new_1 = np.dot(C_inv, np.append(point1, 1))
        new_2 = np.dot(C_inv, np.append(point2, 1))

        new_1 /= np.linalg.norm(new_1)
        new_2 /= np.linalg.norm(new_2)

        spatial_1.append(new_1)
        spatial_2.append(new_2)

    # RANSAC loop
    for _ in range(iterations):
        if func == "lib":
            # sample 4 random pairs of points
            idx = random.sample(range(len(points1)), 4)
            src = np.array([points1[i] for i in idx], np.float32)
            dst = np.array([points2[i] for i in idx], np.float32)

            # find rotation using cv function
            M = cv.getPerspectiveTransform(src=dst, dst=src)
        elif func == "square":
            # sample 2 random pairs of points
            idx = random.sample(range(len(points1)), 2)
            pair_1 = np.array([points1[idx[0]], points2[idx[0]]])
            pair_2 = np.array([points1[idx[1]], points2[idx[1]]])

            # find rotation based on 2 pairs of points
            M = find_transform(pair_1, pair_2)
        elif func == "svd":
            # sample 2 random pairs of points
            idx = random.sample(range(len(points1)), 2)
            src = np.array([spatial_1[i] for i in idx], np.float32)
            dst = np.array([spatial_2[i] for i in idx], np.float32)

            # find rotation using SVD
            M = find_rotation(A=dst, B=src)
        else:
            raise NotImplementedError(f"Unknown func {func}")

        # find inliers
        inliers = []
        distances = []
        if func != "svd":
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
        else:
            for i in range(len(spatial_1)):
                transformed = np.dot(M, spatial_2[i])
                distance = np.linalg.norm(transformed - spatial_1[i])
                distances.append(distance)

            # track the best model (with the least error)
            if sum(distances) < sum(best_distances):
                best_distances = distances
                best_model = M
                best_randoms = idx

    print("\n############\nbest_model")
    print(best_model)
    print(f"selected randoms: {best_randoms}")
    # print(f"len(best_inliers): {len(best_inliers)}")
    print(f"sum(best_distances): {sum(best_distances)}")
    # print(max(best_distances), min(best_distances))

    return best_model

    if func != "svd":
        return best_model

    dict_distances = {i: d for i, d in enumerate(best_distances)}
    best_inliers = sorted(dict_distances.keys(), key=lambda i: dict_distances[i])

    print(dict_distances)
    print(best_inliers)

    inliers_1 = np.array([spatial_1[i] for i in best_inliers[:2]], np.float32)
    inliers_2 = np.array([spatial_2[i] for i in best_inliers[:2]], np.float32)

    R = find_rotation(inliers_2, inliers_1)

    return R


def simulate():
    x_coords = np.random.uniform(-300, 300, 100)
    y_coords = np.random.uniform(-300, 300, 100)
    points1 = np.stack([x_coords, y_coords], axis=-1)
    points2 = points1 @ np.array([[0, 1], [-1, 0]])

    points1 += np.array([640, 360])
    points2 += np.array([640, 360])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(16, 18))
    ax[0].imshow(np.zeros((720, 1280)), cmap="cividis")
    ax[1].imshow(np.zeros((720, 1280)), cmap="cividis")
    ax[0].scatter(points1[:, 0], points1[:, 1], c="white", marker=".")
    ax[1].scatter(points2[:, 0], points2[:, 1], c="white", marker=".")

    f = 1500
    cx, cy = 640, 360
    C = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    C_inv = np.linalg.inv(C)

    R = ransac_loop(points1, points2, iterations=1000, threshold=5, func="svd", C_inv=C_inv)

    transform = C @ R @ C_inv
    # transform = R

    img = cv.imread("simulation-orig.png")
    height, width = img.shape[:2]
    warped_img = cv.warpPerspective(img, transform, (width, height))
    cv.imwrite("simulation-warped.png", warped_img)
    fig.savefig("simulation-frames.png")


if __name__ == "__main__":
    simulate()


    # fig.savefig("points.png")
    # some_M = find_transform(p1, p2)

    # img = cv.imread("test1.png")
    # height, width = img.shape[:2]
    # test_M = np.array([[0.7, -0.7, 100],
    #                    [0.7, 0.7, 700],
    #                    [0, 0, 1]], np.float32)
    # warped_img = cv.warpPerspective(img, test_M, (width, height))
    # cv.imwrite("warped_img.png", warped_img)
