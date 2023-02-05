#!/usr/bin/env python3
import cv2
import numpy as np


def detect_edges(image):
  """Find edge points in a grayscale image.

  Args:
  - image (2D uint8 array): A grayscale image.

  Return:
  - edge_image (2D float array): A heat map where the intensity at each point
      is proportional to the edge magnitude.
  """
  # TODO
  Gx = np.array([[-1, 0, 1],
                 [-2, 0, 2],
                 [-1, 0, 1]], dtype=np.float64)
  Gy = np.array([[1, 2, 1],
                 [0, 0, 0],
                 [-1, -2, -1]], dtype=np.float64)
  edge_image = np.zeros(image.shape)
  img_pad = np.pad(image, 1)
  ex = cv2.filter2D(image, cv2.CV_64F, Gx)
  ey = cv2.filter2D(image, cv2.CV_64F, Gy)
  edge_image = np.sqrt(ex**2 + ey**2).astype(np.float64)
  edge_image *= 255 / np.max(edge_image)
  cv2.imwrite('output/' + "coins_edges_origin.png", edge_image)
  return edge_image

def hough_circles(edge_image, edge_thresh, radius_values):
  """Threshold edge image and calculate the Hough transform accumulator array.

  Args:
  - edge_image (2D float array): An H x W heat map where the intensity at each
      point is proportional to the edge magnitude.
  - edge_thresh (float): A threshold on the edge magnitude values.
  - radius_values (1D int array): An array of R possible radius values.

  Return:
  - thresh_edge_image (2D bool array): Thresholded edge image indicating
      whether each pixel is an edge point or not.
  - accum_array (3D int array): Hough transform accumulator array. Should have
      shape R x H x W.
  """
  #TODO
  angle = np.linspace(-np.pi, np.pi, 360)
  thresh_edge_image = np.zeros(edge_image.shape)
  H, L = edge_image.shape
  thresh_edge_image[edge_image > edge_thresh] = 1
  cv2.imwrite('output/' + "coins_edges.png", thresh_edge_image * 400)
  max_r = np.max(radius_values)
  thresh_edge_image_pad = np.pad(thresh_edge_image, max_r)
  accumulator = np.zeros((H, L, len(radius_values)))
  for i in range(len(radius_values)):
      r = radius_values[i]
      x_center = np.round(r * np.cos(angle)).astype(int)
      y_center = np.round(r * np.sin(angle)).astype(int)
      for k in range(len(angle)):
          accumulator[..., i] += thresh_edge_image_pad[max_r + x_center[k]:max_r + H + x_center[k],
                                  max_r + y_center[k]:max_r + L + y_center[k]]

  fit_items = list(np.unique(accumulator))
  fit_items.sort(reverse=True)
  cnt = 0
  accum_array = []
  while cnt < 100:
      target = fit_items.pop(0)
      fit = list(np.argwhere(accumulator == target))
      cnt += len(fit)
      for c in fit:
          accum_array.append((c, target))

  return thresh_edge_image, accum_array


def find_circles(image, accum_array, radius_values, hough_thresh):
  """Find circles in an image using output from Hough transform.

  Args:
  - image (3D uint8 array): An H x W x 3 BGR color image. Here we use the
      original color image instead of its grayscale version so the circles
      can be drawn in color.
  - accum_array (3D int array): Hough transform accumulator array having shape
      R x H x W.
  - radius_values (1D int array): An array of R radius values.
  - hough_thresh (int): A threshold of votes in the accumulator array.

  Return:
  - circles (list of 3-tuples): A list of circle parameters. Each element
      (r, y, x) represents the radius and the center coordinates of a circle
      found by the program.
  - circle_image (3D uint8 array): A copy of the original image with detected
      circles drawn in color.
  """
  #TODO
  filtered_accum = []
  for c in accum_array:
      if c[1] > hough_thresh:
          filtered_accum.append([c[0][0], c[0][1], radius_values[c[0][2]], c[1]])
  filtered_accum.sort(key=lambda x: x[-1], reverse=True)
  filtered_accum = np.array(filtered_accum)
  while True:
      finish = 1
      for i in range(filtered_accum.shape[0] - 1):
          ref = filtered_accum[i]
          dist = np.sqrt(np.sum((filtered_accum[i + 1:, :2] - ref[:2]) ** 2, axis=1))
          if np.all(dist > ref[2]):
              continue
          else:
              ind = np.ones(filtered_accum.shape[0])
              ind[i + 1:] = dist > ref[2]
              ind = ind.astype(bool)
              filtered_accum = filtered_accum[ind]
              finish = 0
              break
      if finish:
          break

  filtered_accum = list(filtered_accum)
  for c in filtered_accum:
      cv2.circle(image, center=(int(c[1]), int(c[0])), radius=int(c[2]), color=(0, 255, 0), thickness=2)
  cv2.imwrite('output/' + "coins_circles.png", image)

  return [(int(c[2]), int(c[1]), int(c[0])) for c in filtered_accum], image



if __name__ == '__main__':
  #TODO
  img_name = 'coins'
  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edge = detect_edges(gray_image)
  _, accum_array = hough_circles(edge, 120, list(range(20, 41)))
  circles, circle_image = find_circles(gray_image, accum_array, list(range(20, 41)), 100)