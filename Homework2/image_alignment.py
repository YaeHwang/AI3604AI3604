#!/usr/bin/env python3
import cv2
import utils as utils
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from typing import List, Tuple
# import matplotlib.pyplot as plt

def detect_blobs(image):
    """Laplacian blob detector.
    Args:
    - image (2D float64 array): A grayscale image.
    Returns:
    - corners (list of 2-tuples): A list of 2-tuples representing the locations
        of detected blobs. Each tuple contains the (x, y) coordinates of a
        pixel, which can be indexed by image[y, x].
    - scales (list of floats): A list of floats representing the scales of
        detected blobs. Has the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.
    """
    sigmas = []
    gaussian_images = []

    for k in range(6):
        if k == 0:
            sigma = 5
        else:
            sigma = 5 * 1.5 ** k
        sigmas.append(sigma)
        filtered = gaussian_filter(image, sigma)
        gaussian_images.append(filtered)
     
    local_maximums = maximum_filter(np.diff(gaussian_images, n=1, axis=0), size=(3, 3, 3))
    blobs = np.max(local_maximums, axis=0)
    th = np.quantile(blobs, 0.9)
    sizes = np.argmax(local_maximums, axis=0)
    blob_bin: np.ndarray = np.where(blobs >= th, 255, 0)
    labeled_blobs = utils.DivideImg(blob_bin).object_segmentation()
    attrs = utils.GetAttr(labeled_blobs).get_attrs()

    corners = []
    scales = []
    orientations = []

    for attr in attrs:        
        x, y = attr['position']['x'], attr['position']['y']
        corners.append((x, y))
        scales.append(sigmas[sizes[int(y), int(x)]])
        orientations.append(attr['orientation'])
    return corners, scales, orientations


def compute_descriptors(image, corners, scales, orientations):
    """Compute descriptors for corners at specified scales.

    Args:
    - image (2d float64 array): A grayscale image.
    - corners (list of 2-tuples): A list of (x, y) coordinates.
    - scales (list of floats): A list of scales corresponding to the corners.
        Must have the same length as `corners`.
    - orientations (list of floats): A list of floats representing the dominant
        orientation of the blobs.

    Returns:
    - descriptors (list of 1d array): A list of desciptors for each corner.
        Each element is an 1d array of length 128.
    """
    if len(corners) != len(scales) or len(corners) != len(orientations):
        raise ValueError(
            '`corners`, `scales` and `orientations` must all have the same length.')
    descriptor = [] 

    for (x, y), scale, theta in zip(corners, scales, orientations):
        x, y, scale = int(x), int(y), int(scale)
        BLOCK_SIZE = 4 * int(scale)
        NEIGHBOUR_SIZE = 4 * BLOCK_SIZE
        offset = 2 * BLOCK_SIZE

        if x >= image.shape[1] - offset or x < offset or y >= image.shape[0] - offset or y < offset:          
            descriptor.append(None)
        else:
            region_image = image[y-offset: y+offset, x-offset: x+offset]
            

            grad_x, grad_y = np.gradient(region_image)

            grad_abs = np.sqrt(np.square(grad_x)+np.square(grad_y))
            kernel = cv2.getGaussianKernel(NEIGHBOUR_SIZE, 2*scale).T * cv2.getGaussianKernel(NEIGHBOUR_SIZE, 2*scale)
            grad_abs = np.multiply(grad_abs, kernel)
            grad_angle = np.arctan2(grad_x, grad_y) - theta 



            histrogram_list = []
            block_num = int(NEIGHBOUR_SIZE/BLOCK_SIZE)
            for i in range(block_num):
                for j in range(block_num):
                    hist = np.zeros(8) 

                    block_grad_angle = grad_angle[i*BLOCK_SIZE: (i+1)*BLOCK_SIZE,j*BLOCK_SIZE: (j+1)*BLOCK_SIZE]
                    block_grad_abs = grad_abs[i*BLOCK_SIZE: (i+1)*BLOCK_SIZE,j*BLOCK_SIZE: (j+1)*BLOCK_SIZE]
                    
                    for angle, grad in zip(block_grad_angle.reshape(-1), block_grad_abs.reshape(-1)):
                        if angle < 0:
                            angle += 2 * np.pi
                        

                        angle_bin = (angle / (np.pi / 4))
                        higher_bin_weight = angle_bin - int(angle_bin)

                        hist[int(angle_bin) % 8] += (1 - higher_bin_weight) * grad
                        hist[(int(angle_bin) + 1) % 8] += higher_bin_weight * grad
                    histrogram_list.append(hist)

            histrogram_list = np.abs(np.array(histrogram_list).reshape(-1))
            histrogram_list = histrogram_list / np.linalg.norm(histrogram_list, ord=2)
            
            histrogram_list = np.clip(histrogram_list, -0.3, 0.3)
            histrogram_list = histrogram_list / np.linalg.norm(histrogram_list, ord=2)

            descriptor.append(histrogram_list)

    return descriptor


def match_descriptors(descriptors1, descriptors2):
    """Match descriptors based on their L2-distance and the "ratio test".

    Args:
    - descriptors1 (list of 1d arrays):
    - descriptors2 (list of 1d arrays):

    Returns:
    - matches (list of 2-tuples): A list of 2-tuples representing the matching
        indices. Each tuple contains two integer indices. For example, tuple
        (0, 42) indicates that corners1[0] is matched to corners2[42].
    """
    
    matches: List[Tuple] = []
    dist_mat = np.zeros((len(descriptors1), len(descriptors2)))
    for i, d1 in enumerate(descriptors1):
        for j, d2 in enumerate(descriptors2):
            if d1 is None or d2 is None:
                dist_mat[i, j] = 1e8
            else:
                dist_mat[i, j] = np.linalg.norm((d1 - d2), ord=2)
        candidates = np.argsort(dist_mat[i, :])
        best, second = candidates[:2]
        if dist_mat[i, best] / dist_mat[i, second] <= 0.7:
            matches.append((i, best))

    return matches


def draw_matches(image1, image2, corners1, corners2, matches,
                 outlier_labels=None):
    """Draw matched corners between images.

    Args:
    - matches (list of 2-tuples)
    - image1 (3D uint8 array): A color image having shape (H1, W1, 3).
    - image2 (3D uint8 array): A color image having shape (H2, W2, 3).
    - corners1 (list of 2-tuples)
    - corners2 (list of 2-tuples)
    - outlier_labels (list of bool)

    Returns:
    - match_image (3D uint8 array): A color image having shape
        (max(H1, H2), W1 + W2, 3).
    """
    if outlier_labels is None:
        outlier_labels = np.ones(len(matches))
    colors = [
        (0,0,255) if is_outlier else (255,0,0)
        for is_outlier in outlier_labels]

    W1 = image1.shape[1]
    H1, H2 = image1.shape[0], image2.shape[0]
    H = max(H1, H2)

    image1 = np.pad(image1, [(0, max(0, H-image1.shape[0])), (0, 0), (0, 0)])
    image2 = np.pad(image2, [(0, max(0, H-image2.shape[0])), (0, 0), (0, 0)])

    match_image = np.concatenate((image1, image2), axis=1)

    for color, (c1idx, c2idx) in zip(colors, matches):
        x1, y1 = list(map(int, corners1[c1idx]))
        x2, y2 = list(map(int, corners2[c2idx]))
        x2 += W1
        match_image = cv2.line(match_image,(x1, y1),(x2, y2),color=color,thickness=2)

    return match_image


def compute_affine_xform(corners1, corners2, matches):
    """Compute affine transformation given matched feature locations.

    Args:
    - corners1 (list of 2-tuples)
    - corners1 (list of 2-tuples)
    - matches (list of 2-tuples)

    Returns:
    - xform (2D float64 array): A 3x3 matrix representing the affine
        transformation that maps coordinates in image1 to the corresponding
        coordinates in image2.
    - outlier_labels (list of bool): A list of Boolean values indicating
        whether the corresponding match in `matches` is an outlier or not.
        For example, if `matches[42]` is determined as an outlier match
        after RANSAC, then `outlier_labels[42]` should have value `True`.
    """
    max_inliner = 0
    xform = None
    outliners = None
    u = []  
    v = []  
    for a, b in matches:
        u.append(corners1[a])
        v.append(corners2[b])
    u, v = np.array(u) + 200, np.array(v) + 200
    u = np.concatenate([u, np.ones_like(u[:, :1])], 1)
    v = np.concatenate([v, np.ones_like(v[:, :1])], 1)
    for _ in range(100):
        rand_samp = np.random.randint(len(u), size=[6])
        aff = np.linalg.lstsq(u[rand_samp], v[rand_samp], rcond=1e-5)[0]
        inliners = np.linalg.norm(u.dot(aff) - v, axis=1) < 12
        if inliners.sum() > max_inliner:
            max_inliner = inliners.sum()
            xform = np.linalg.lstsq(u[inliners], v[inliners], rcond=1e-5)[0]
            outliners = ~inliners
    return xform, outliners


def stitch_images(image1, image2, xform):
    """Stitch two matched images given the transformation between them.

    Args:
    - image1 (3D uint8 array): A color image.
    - image2 (3D uint8 array): A color image.
    - xform (2D float64 array): A 3x3 matrix representing the transformation
        between image1 and image2. This transformation should map coordinates
        in image1 to the corresponding coordinates in image2.

    Returns:
    - image_stitched (3D uint8 array)
    """
    image1 = np.pad(image1, [(200, 200), (200, 200), (0, 0)])
    image2 = np.pad(image2, [(200, 200), (200, 200), (0, 0)])
    image_warped: np.ndarray = cv2.warpAffine(
        image1,
        xform[:2],
        image2.shape[:2][::-1])
    image_stitched = image_warped.astype(np.float32)\
        + image2.astype(np.float32)
    image_stitched[(image2 > 1e-3) & (image_warped > 1e-3)] /= 2.0
    return image_stitched.astype(np.uint8)


def main():
    img_name = 'leuven'
    img_id1 = 1
    img_id2 = 3
    img_path1 = f'data/{img_name}{img_id1}.png'
    img_path2 = f'data/{img_name}{img_id2}.png'

    img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) / 255.0
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) / 255.0

    # TODO
    print(f'Matching {img_name} {img_id1} and {img_id2}')

    c1, s1, ori1 = detect_blobs(gray1)
    c2, s2, ori2 = detect_blobs(gray2)

    print('Num of Blobs:', len(c1), len(c2))

    d1 = compute_descriptors(gray1, c1, s1, ori1)
    d2 = compute_descriptors(gray2, c2, s2, ori2)

    print('Num of Descriptors', len(d1), len(d2))

    matches = match_descriptors(d1, d2)
    print('Matches:', matches)

    xform, outliers = compute_affine_xform(c1, c2, matches)

    visualization = draw_matches(img1, img2, c1, c2, matches, outliers)
    cv2.imwrite(
        f'./data/{img_name}_{img_id1}{img_id2}_match.png',
        visualization.astype(np.uint8))

    if xform is None:
        print('Matching failed. Exiting.')
        return

    stitched = stitch_images(img1, img2, xform)
    cv2.imwrite(
        f'./data/{img_name}_{img_id1}{img_id2}_stitch.png',
        stitched)


if __name__ == '__main__':
    main()
