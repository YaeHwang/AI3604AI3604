#!/usr/bin/env python3
import cv2
import numpy as np
import sys

from numpy.typing import _128Bit


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0
  binary_image = np.where(gray_image >= thresh_val, 255, 0)
  return binary_image

def label(binary_image):
  # TODO
  Row, Column = binary_image.shape
  labeled_image = np.zeros(binary_image.shape)
  parent = np.zeros(binary_image.size, dtype=int)
  record = {}
  mark = 1
  "pass 1"
  for r in range(binary_image.shape[0]):
    for c in range(binary_image.shape[1]):
      if binary_image[r, c] == 0:
        continue
      neighbors = get_neighbors((r, c), binary_image)
      if len(neighbors) == 0:
        labeled_image[r, c] = mark
        record[r * Column + c] = mark
        mark += 1
      else:
        labels = [labeled_image[p[0], p[1]] for p in neighbors]
        parent_ind = [p[0] * Column + p[1] for p in neighbors]
        min = np.argmin(labels)
        parent[r * Column + c] = parent_ind[min]
        labeled_image[r, c] = labels[min]
        for i in range(len(labels)):
          if labels[i] != labels[min]:
            union(parent_ind[min], parent_ind[i], parent)
  "pass 2"
  for r in range(binary_image.shape[0]):
    for c in range(binary_image.shape[1]):
      if binary_image[r, c] > 0:
        labeled_image[r, c] = record[find(parent, r * Column + c)]
  intensity = np.unique(labeled_image)
  for i in range(len(intensity)):
      if np.sum(labeled_image == intensity[i]) < 50:  
          labeled_image[labeled_image == intensity[i]] = 0
          continue
      labeled_image[labeled_image == intensity[i]] = i
  labeled_image = labeled_image * 255 / np.max(labeled_image)
  return labeled_image

def get_neighbors(p, img):
    r, c = p
    neighbors = []
    if r > 0 and img[r - 1, c] > 0:
        neighbors.append((r - 1, c))
    if c > 0 and img[r, c - 1] > 0:
        neighbors.append((r, c - 1))
    return neighbors


def find(parent, x):
    j = x
    while parent[j] != 0:
        j = parent[j]
    return j


def union(x, y, parent):
    j = find(parent, x)
    k = find(parent, y)
    if j != k:
        parent[k] = j


def get_attribute(labeled_image):
  # TODO
  orientation = labeled_image.copy()
  Row, Column = labeled_image.shape
  attribute_list = []
  intensities = list(np.unique(labeled_image))
  intensities.pop(0)
  for i in range(len(intensities)):
      img_with_one_obj = np.zeros(labeled_image.shape)
      img_with_one_obj[labeled_image == intensities[i]] = 1
      attribute = {'position': {}}
      obj_points = np.argwhere(img_with_one_obj == 1)
      x = obj_points[:, 1]
      y = Row-1-obj_points[:, 0]
      x_ave = np.mean(x)
      y_ave = np.mean(y)
      attribute['position']['x'] = x_ave
      attribute['position']['y'] = y_ave
      x_1 = x - x_ave
      y_1 = y - y_ave
      a = np.sum(x_1 ** 2)
      b = 2 * np.sum(x_1 * y_1)
      c = np.sum(y_1 ** 2)
      theta_1 = np.arctan(b/(a-c)) / 2
      theta_2 = theta_1 + np.pi/2

      e_1 = a * (np.sin(theta_1) ** 2) - b * np.sin(theta_1) * np.cos(theta_1) + c * (np.cos(theta_1) ** 2)
      e_2 = a * (np.sin(theta_2) ** 2) - b * np.sin(theta_2) * np.cos(theta_2) + c * (np.cos(theta_2) ** 2)
      if e_1 < e_2:
          attribute['orientation'] = theta_1
          attribute['roundedness'] = e_1 / e_2
      else:
          attribute['orientation'] = theta_2
          attribute['roundedness'] = e_2 / e_1

      attribute_list.append(attribute)

      theta = attribute['orientation']
      x = np.array(range(0, 50))
      for k in range(len(x)):
          if round(y_ave + x[k] * np.tan(theta)) < 0 or round(y_ave + x[k] * np.tan(theta)) > Row-1:
              continue
          orientation[Row-1-round(y_ave + x[k] * np.tan(theta)), round(x_ave + x[k])] = 255
  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  # print(attribute_list)
  for i in attribute_list:
        print(i)


if __name__ == '__main__':
  # main(sys.argv[1:])
  # main(['many_objects_1', 128])
  main(['many_objects_2', 128])