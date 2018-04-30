import numpy as np
from Sample import Sample
import os
from skimage import io
import skimage.measure
import progressbar
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


class KNN:

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([]).astype(int)
        self.knn = KNeighborsClassifier()

    @staticmethod
    def get_result(file_name):
        return [1] if file_name.split(".")[0][-1] == "y" else [0]

    def train(self):
        try:
            img_list = os.listdir(Sample.sample_path)
            assert len(img_list) > 0
        except (FileNotFoundError, AssertionError):
            print("KNN.train(): No samples found :(")
            return

        print("KNN.train(): training started")
        rand_order_list = np.random.permutation(img_list)
        bar = progressbar.ProgressBar(maxval=len(rand_order_list),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for i, file_name in enumerate(rand_order_list):
            img = io.imread(Sample.sample_path + file_name, as_grey=True)
            self.x = np.append(self.x, skimage.measure.moments_hu(img), axis=0)
            self.y = np.append(self.y, KNN.get_result(file_name), axis=0)
            bar.update(i+1)

        bar.finish()
        self.x = np.reshape(self.x, (int(len(self.x)/7), 7))
        self.knn.fit(self.x, self.y)
        print("KNN.train(): finished! Classifier is ready to work!")
"""
    def predict(self, file, sample_size):
        sample_half = int(sample_size/2)
        img = Image.open(file)
        # print("original", img.size)
        w, h = img.size
        img = ImageOps.expand(img, border=sample_half, fill='black')
        # print("expand", img.size)
        border_file = file.split(".")[0] + "_border." + file.split(".")[1]
        img.save(border_file)

        img = io.imread(border_file, as_grey=True)
        result = np.ones((h, w))/2
        # print(result.shape)

        #print(self.knn.predict([[1, 1, 1, 1, 1, 2, 2]]))
        #print(self.knn.predict(np.array([1, 2, 3, 4, 5, 6, 7]).reshape(1, -1)))

        #iter_x = img.shape[0] - 2*sample_half
        #iter_y = img.shape[1] - 2*sample_half
        bar = progressbar.ProgressBar(maxval=h*w,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        bar.update(0)
        counter = 0

        #print("iterate over", iter_x, iter_y)
        for i in np.arange(0, h):
            for j in np.arange(0, w):
                hu = skimage.measure.moments_hu(img[i:i+sample_size, j:j+sample_size])
                #mi = i + sample_half
                #mj = j + sample_half
                #print(hu)
                #result[mi][mj] = self.knn.predict([hu])
                result[i][j] = self.knn.predict([hu])
                counter += 1
                bar.update(counter)
        bar.finish()
        np.set_printoptions(threshold=np.nan)
        print(result)
        #result = result[sample_half:, sample_half:]
        io.imsave("01_hhh.jpg", result)
        #result = result * 255
        print("maxval", h*w, "counter", counter)

"""