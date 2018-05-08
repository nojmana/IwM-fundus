import numpy as np
from scipy._lib.six import reduce
from skimage import filters
from PIL import Image
import cv2
import scipy
import operator


class ImageProcessing:
    @staticmethod
    def gaussian_filter(picture, sigma):
        return scipy.ndimage.gaussian_filter(picture, sigma=sigma)

    @staticmethod
    def median_filter(picture, mask):
        return filters.rank.median(picture, np.ones([mask, mask], dtype=np.uint8))

    @staticmethod
    def hist_normalization(picture):
        histogram = Image.fromarray(picture).convert("L").histogram()
        lut = []
        for b in range(0, len(histogram), 256):
            step = reduce(operator.add, histogram[b:b + 256]) / 255
            n = 0
            for i in range(256):
                lut.append(n / step)
                n = n + histogram[i + b]
        return np.asarray(Image.fromarray(picture).point(lut * 1))

    def process_picture(self, input_picture):
        output_picture = cv2.cvtColor(input_picture, cv2.COLOR_RGB2GRAY)
        output_picture = self.median_filter(output_picture, 3)
        output_picture = cv2.Canny(output_picture, 20, 50)
        output_picture = self.gaussian_filter(output_picture, 2)
        output_picture = self.hist_normalization(output_picture)
        output_contours, contours, hierarchy = cv2.findContours(output_picture, cv2.RETR_TREE,
                                                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(input_picture, contours, -1, (255, 255, 255), 3)
        return output_picture
