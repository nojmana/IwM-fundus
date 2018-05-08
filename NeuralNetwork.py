from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import skimage
import progressbar
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:
    n_moments_hu = 7

    def __init__(self):
        self.x = np.array([])
        self.y = np.array([]).astype(int)
        self.model = Sequential()
        self.hu = np.array([])

    @staticmethod
    def expand(img, border, value=0):
        result = np.ones((img.shape[0]+2*border, img.shape[1]+2*border)) * value
        result[border:img.shape[0]+border, border:img.shape[1]+border] = img
        return result

    def learning_curve(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def train(self, file, validation_split, epochs):
        try:
            dataframe = pd.read_csv(file)
        except FileNotFoundError:
            print("NeuralNetworks.train(): No samples found :(")
            return
        dataframe = shuffle(dataframe)
        dataset = dataframe.values

        x = dataset[:, :7].astype(float)
        y = dataset[:, -1].astype(int)

        self.model.add(Dense(30, input_dim=7, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(x, y, validation_split=0.1, epochs=epochs, verbose=2)
        self.learning_curve(history)

    def predict(self, img, sample_size):
        img = skimage.color.rgb2gray(img)
        h, w = img.shape
        half_ss = int(sample_size/2)

        img = NeuralNetwork.expand(img, half_ss)
        hu = np.zeros((h, w, NeuralNetwork.n_moments_hu))

        print('\nCounting moments hu in progress...')
        bar = progressbar.ProgressBar(maxval=h,
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        counter = 0

        for i in np.arange(h):
            bar.update(counter)
            counter += 1
            for j in np.arange(w):
                hu[i, j] = skimage.measure.moments_hu(img[i:i + sample_size, j:j + sample_size])
        bar.finish()

        print('\nPrediction in progress...')
        result = np.zeros((h, w))
        counter = 0
        for i in np.arange(h):
            bar.update(counter)
            counter += 1

            row_result = self.model.predict_classes(hu[i])
            result[i] = np.concatenate(row_result)
        bar.finish()

        result = (result * 255).astype(np.uint8)
        print(result)
        print('Prediction finished!')
        return result
