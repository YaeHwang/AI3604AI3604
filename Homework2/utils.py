import cv2
import numpy as np
import sys
# import matplotlib.pyplot as plt


class GetAttr():
    def __init__(self, labeled_image: np.ndarray):
        self.image = labeled_image
        self.shape = labeled_image.shape
        self.attrs = []
        self.objs = set([*self.image.flatten()])

    def get_attrs(self):
        for obj_intensity in self.objs:
            if obj_intensity == 0:
                continue
            current_attrs = {}
            current_obj = self._get_current_obj(obj_intensity)
            pos_x, pos_y = self.get_position(current_obj)
            pos_dict = {}
            pos_dict['x'] = pos_x
            pos_dict['y'] = pos_y
            current_attrs['position'] = pos_dict
            second_moments = self._get_centered_2nd_moments(
                current_obj, (pos_x, pos_y))
            orientation = self.get_orientation(second_moments)
            current_attrs['orientation'] = orientation
            rounedness = self.get_roundedness(second_moments, orientation)
            current_attrs['roundedness'] = rounedness

            self.attrs.append(current_attrs)

        return self.attrs

    def _get_current_obj(self, obj_intensity) -> np.ndarray:
        current_obj = np.where(self.image == obj_intensity, 1, 0)
        return current_obj

    def get_position(self, current_obj: np.ndarray):
        A = current_obj.sum()
        y_size, x_size = self.shape

        x_axis = np.linspace(0, x_size, x_size, endpoint=False)
        x = np.multiply(current_obj, np.expand_dims(x_axis, 0)).sum() / A

        y_axis = np.linspace(0, y_size, y_size, endpoint=False)
        y = np.multiply(current_obj, np.expand_dims(y_axis, -1)).sum() / A

        return (x, y)

    def _get_centered_2nd_moments(self, current_obj: np.ndarray, position):
        x, y = position
        y_size, x_size = self.shape

        x_centered = np.linspace(0, x_size, x_size, endpoint=False) - x
        xx: np.ndarray = np.expand_dims(x_centered ** 2, 0)
        y_centered = np.linspace(0, y_size, y_size, endpoint=False) - y
        yy: np.ndarray = np.expand_dims(y_centered ** 2, -1)
        xy = np.multiply(
            np.expand_dims(x_centered, 0), np.expand_dims(y_centered, -1))

        a = np.multiply(xx, current_obj).sum()
        b = 2 * np.multiply(xy, current_obj).sum()
        c = np.multiply(yy, current_obj).sum()

        return (a, b, c)

    def get_orientation(self, second_moments):
        a, b, c = second_moments
        if a == c:
            return 0.5 * np.pi
        theta = 0.5 * np.arctan(b / (a-c))
        theta_prime = theta + 0.5 * np.pi
        e_theta = self._get_bigE(second_moments, theta)
        e_theta_prime = self._get_bigE(second_moments, theta_prime)
        orientation = theta if e_theta < e_theta_prime else theta_prime

        return orientation

    def _get_bigE(self, moments, theta):
        a, b, c = moments
        E = a * (np.sin(theta)**2)\
            - b * (np.sin(theta)*np.cos(theta))\
            + c * (np.cos(theta)**2)
        return E

    def _get_emax_emin(self, moments, orientation):
        theta = orientation
        theta_prime = orientation + 0.5 * np.pi
        e_min = self._get_bigE(moments, theta)
        e_max = self._get_bigE(moments, theta_prime)

        return e_min, e_max

    def get_roundedness(self, moments, orientation):
        e_min, e_max = self._get_emax_emin(moments, orientation)
        return e_min / e_max


class UnionFindSet():
    def __init__(self, elements):
        self.uf_set = {}
        for e in elements:
            self.uf_set[e] = e

    def find(self, x):
        if self.uf_set[x] != x:
            self.uf_set[x] = self.find(self.uf_set[x])
            return self.uf_set[x]
        return x

    def union(self, x, y):
        if self.find(x) == self.find(y):
            return
        else:
            self.uf_set[self.find(x)] = self.uf_set[self.find(y)]
        return


class DivideImg():
    """Implements the sequential labeling algorithm.

    Args:
      image:
        Input binary image, as a numpy ndarray.
    """

    def __init__(self, image: np.ndarray):
        self.BG = 0
        self.FG = 255
        self.current_label = 0
        self.image = image
        self.labeled_image = np.zeros_like(image)
        elements = []
        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                elements.append((x, y))
        self.equivs = UnionFindSet(elements)

    def object_segmentation(self) -> np.ndarray:
        # first pass
        for x, columns in enumerate(self.image):
            for y, pixel in enumerate(columns):
                self._get_label(x, y, pixel)

        # second pass
        objs = {}
        for x, columns in enumerate(self.image):
            for y, pixel in enumerate(columns):
                label = self.equivs.find((x, y))
                if label not in objs:
                    objs[label] = self._assign_new_label()

        print('Num of objects (including BG):', len(objs.keys()))
        if len(objs.keys()) > 255:
            raise ValueError('This implementation only supports < 255 objs')

        # third pass for coloring
        color_interval = np.floor(255 / self.current_label)
        for x, columns in enumerate(self.image):
            for y, pixel in enumerate(columns):
                label = self.equivs.find((x, y))
                self.labeled_image[x, y] = objs[label] * color_interval

        return self.labeled_image

    def _assign_new_label(self):
        self.current_label += 1
        return self.current_label - 1

    def _get_label(self, x, y, pixel):
        if x == 0 or y == 0:
            self.equivs.union((x, y), (0, 0))
            return

        if pixel == self.BG:
            self.equivs.union((x, y), (0, 0))
            return


        elif self.image[x-1, y-1] == self.FG:

            self.equivs.union((x, y), (x-1, y-1))
            return
        elif self._is_only_fg_pixel(x, y):

            return
        elif self._is_neighbor_x_fg(x, y):

            self.equivs.union((x, y), (x-1, y))
            return
        elif self._is_neighbor_y_fg(x, y):

            self.equivs.union((x, y), (x, y-1))
            return
        elif self._is_neighbor_both_fg(x, y):

            xlabel = self.equivs.find((x, y-1))
            ylabel = self.equivs.find((x-1, y))
            if xlabel != ylabel:
                self.equivs.union((x, y-1), (x-1, y))
            self.equivs.union((x, y), (x-1, y))
            return
        else:
            raise ValueError(f'Unexpected input {pixel}')

    def _is_only_fg_pixel(self, x, y):
        return (self.image[x-1, y-1] == self.BG
                and self.image[x, y-1] == self.BG
                and self.image[x-1, y] == self.BG)

    def _is_neighbor_x_fg(self, x, y):
        return (self.image[x-1, y-1] == self.BG
                and self.image[x-1, y] == self.FG
                and self.image[x, y-1] == self.BG)

    def _is_neighbor_y_fg(self, x, y):
        return (self.image[x-1, y-1] == self.BG
                and self.image[x-1, y] == self.BG
                and self.image[x, y-1] == self.FG)

    def _is_neighbor_both_fg(self, x, y):
        return (self.image[x-1, y-1] == self.BG
                and self.image[x-1, y] == self.FG
                and self.image[x, y-1] == self.FG)


def binarize(gray_image, thresh_val):

    binary_image = np.where(gray_image >= thresh_val, 255, 0)
    return binary_image


def label(binary_image):
    divider = DivideImg(binary_image)
    labeled_image = divider.object_segmentation()
    return labeled_image


def get_attribute(labeled_image):
    attr_counter = GetAttr(labeled_image)
    attribute_list = attr_counter.get_attrs()
    return attribute_list


def annotate_attributes(labeled_image, attribute_list, c=1):
    ARROW_LENGTH = 100
    annotated_image = np.array(labeled_image)

    def invert_y(y):
        return y

    for attr_dict in attribute_list:
        x = attr_dict['position']['x']
        y = attr_dict['position']['y']
        orientation = attr_dict['orientation']

        xe = x + np.cos(orientation) * ARROW_LENGTH
        ye = y + np.sin(orientation) * ARROW_LENGTH
        annotated_image = cv2.arrowedLine(
            annotated_image,
            (int(x), int(invert_y(y))),
            (int(xe), int(invert_y(ye))),
            (c, 0, 0),
            thickness=2
        )
        annotated_image = cv2.putText(
            annotated_image,
            f'{np.rad2deg(attr_dict["orientation"]):.2f}',
            (int(xe), int(invert_y(ye))),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (c, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA)

    return annotated_image

