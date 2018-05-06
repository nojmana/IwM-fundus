import numpy as np
import pandas as pd
from skimage import io
import skimage.measure
import progressbar
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageOps
from sklearn.utils import shuffle


class KNN:

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([]).astype(int)
        self.knn = KNeighborsClassifier()

    @staticmethod
    def get_result(file_name):
        return [1] if file_name.split(".")[0][-1] == "y" else [0]

    def train(self, file):
        try:
            data_frame = pd.read_csv(file)
        except FileNotFoundError:
            print("KNN.train(): No samples found :(")
            return
        data_frame = shuffle(data_frame)
        data_set = data_frame.values
        x = data_set[:, :7].astype(float)
        y = data_set[:, -1].astype(int)

        print("KNN.train(): training started")
        self.knn.fit(x, y)
        print("KNN.train(): finished! Classifier is ready to work!")

    def predict(self, file, sample_size):
        sample_half = int(sample_size/2)
        img = Image.open(file)
        w, h = img.size
        img = ImageOps.expand(img, border=sample_half, fill='black')
        border_file = file.split(".")[0] + "_border." + file.split(".")[1]
        img.save(border_file)

        img = io.imread(border_file, as_grey=True)
        result = np.ones((h, w))/2

        bar = progressbar.ProgressBar(maxval=h*w,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        bar.update(0)
        counter = 0

        for i in np.arange(0, h):
            for j in np.arange(0, w):
                hu = skimage.measure.moments_hu(img[i:i+sample_size, j:j+sample_size])
                result[i][j] = self.knn.predict([hu])
                counter += 1
                bar.update(counter)
        bar.finish()
        np.set_printoptions(threshold=np.nan)
        io.imsave("KNN.jpg", result)
        return (result * 255).astype(np.uint8)
