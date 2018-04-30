from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os
from Sample import Sample
import skimage
from skimage import io


class NeuralNetwork:

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([]).astype(int)
        self.model = Sequential()
        self.model.add(Dense(30, input_dim=7, init='uniform', activation='relu'))
        self.model.add(Dense(30, init='uniform', activation='relu'))
        self.model.add(Dense(1, init='uniform', activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    @staticmethod
    def get_result(file_name):
        return [1] if file_name.split(".")[0][-1] == "y" else [0]

    @staticmethod
    def expand(img, border):
        #print(img.shape)
        result = np.zeros((img.shape[0]+2*border, img.shape[1]+2*border))
        result[border:img.shape[0]+border, border:img.shape[1]+border] = img
        return result

    def train(self):
        try:
            img_list = os.listdir(Sample.sample_path)
            assert len(img_list) > 0
        except (FileNotFoundError, AssertionError):
            print("NeuralNetworks.train(): No samples found :(")
            return

        rand_order_list = np.random.permutation(img_list)
        for i, file_name in enumerate(rand_order_list):
            img = io.imread(Sample.sample_path + file_name, as_grey=True)
            self.x = np.append(self.x, skimage.measure.moments_hu(img), axis=0)
            self.y = np.append(self.y, NeuralNetwork.get_result(file_name), axis=0)
        self.x = np.reshape(self.x, (int(len(self.x) / 7), 7))
        self.model.fit(self.x, self.y, epochs=150, batch_size=10)

"""
    def predict(self, file, sample_size):
        border = int(sample_size/2)
        img = io.imread(file, as_grey=True)
        img = NeuralNetwork.expand(img, border)

        hu = np.array([]) # np.zeros(img.shape)

        bar = progressbar.ProgressBar(maxval=img.shape[0],
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        counter = 0

        for i in np.arange(img.shape[0]):
            counter += 1
            bar.update(counter)
            for j in np.arange(img.shape[1]):
                hu = np.append(hu, skimage.measure.moments_hu(img[i:i + sample_size, j:j + sample_size]))
        hu = np.reshape(hu, img.shape[0], img.shape[1])
        print(hu)

                #hu = skimage.measure.moments_hu(img[i:i + sample_size, j:j + sample_size])
                #result[i][j] =

"""
""" def train(self):
        dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
        # split into input (X) and output (Y) variables
        x = dataset[:, 0:8]
        y = dataset[:, 8]
        # print(x, y)
        self.model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
        self.model.add(Dense(8, init='uniform', activation='relu'))
        self.model.add(Dense(1, init='uniform', activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x, y, epochs=150, batch_size=10)
        scores = self.model.evaluate(x, y)

    def predict(self):
        dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
        # split into input (X) and output (Y) variables
        x = dataset[:, 0:8]
        predictions = self.model.predict(x)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        print(rounded)
        """