import os

import Image_processing
from Sample import Sample
from KNN import KNN
from NeuralNetwork import NeuralNetwork
from Analysis import Analysis

from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Frame, Button, Label

from PIL import Image, ImageTk
from skimage import io
import numpy as np


class MainWindow(Frame):

    sample_size = 21
    number_of_pics = 1200
    n_file_samples = 15
    sample_path = 'samples.csv'
    predicted_path = 'predicted.jpg'

    def __init__(self, root, file):
        super().__init__()
        self.root = root
        self.file = file

        self.quit_button = Button(self, text="Quit", command=self.quit)
        self.browse_button = Button(self, text="Browse file", command=self.browse)
        self.sample_button = Button(self, text="Create samples", command=self.sample)
        self.knn_learn_button = Button(self, text="KNN learn", command=self.knn_learn)
        self.knn_predict_button = Button(self, text="KNN predict", command=self.knn_predict)
        self.nn_train_button = Button(self, text="Neural Network train", command=self.nn_train)
        self.nn_predict_button = Button(self, text="Neural Network predict", command=self.nn_predict)
        self.nn_tp_button = Button(self, text="NN train and predict", command=self.nn_tp)
        self.nn_analysis_button = Button(self, text="NN analysis", command=self.nn_analysis)
        self.init_ui()

        self.input_picture = io.imread(file)
        self.display_picture(Image.fromarray(self.input_picture), 'input')
        self.image_processing = Image_processing.ImageProcessing()
        self.output_picture = self.image_processing.process_picture(self.input_picture)
        self.display_picture(Image.fromarray(self.output_picture), 'output')
        self.predicted = None
        if os.path.isfile(self.file.split("/healthy")[0] + "/manual/" + self.file.split(".")[0][-4:] + '.tif'):
            self.predicted_original = io.imread(self.file.split("/healthy")[0] + "/manual/" + self.file.split(".")[0][-4:] + '.tif')
        else:
            self.predicted_original = None
        self.knn = None
        self.nn = None
        self.nn_analysis = None

    def init_ui(self):
        self.master.title("Detection of fundus blood vessels")
        self.pack(fill=BOTH, expand=1)
        self.center_window()

        self.quit_button.place(x=985, y=15)
        self.browse_button.place(x=900, y=15)
        self.sample_button.place(x=800, y=15)
        self.knn_learn_button.place(x=720, y=15)
        self.knn_predict_button.place(x=640, y=15)
        self.nn_train_button.place(x=500, y=15)
        self.nn_predict_button.place(x=350, y=15)
        self.nn_tp_button.place(x=220, y=15)
        self.nn_analysis_button.place(x=140, y=15)

    def center_window(self):
        w = 1090
        h = 580

        x = (self.master.winfo_screenwidth() - w) / 2
        y = (self.master.winfo_screenheight() - h) / 2
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def sample(self):
        Sample.generate_csv(MainWindow.sample_size, MainWindow.number_of_pics, MainWindow.n_file_samples,
                            MainWindow.sample_path)
        print("Creating samples finished!")

    def knn_learn(self):
        self.knn = KNN()
        self.knn.train(MainWindow.sample_path)

    def knn_predict(self):
        self.result = self.knn.predict(self.file, sample_size=MainWindow.sample_size)
        self.display_picture(Image.fromarray(self.result), 'output')

    def nn_train(self, validation_split=0.33, epochs=50):
        self.nn = NeuralNetwork()
        self.nn.train(MainWindow.sample_path, validation_split, epochs)

    def nn_predict(self):
        if self.nn is None:
            print("Neural network has to be trained before prediction!")
            return
        self.predicted = self.nn.predict(self.input_picture, sample_size=MainWindow.sample_size)
        io.imsave(MainWindow.predicted_path, self.predicted)
        self.display_picture(Image.fromarray(self.predicted), 'output')

    def nn_analysis(self):
        if not os.path.isfile(self.file.split("/healthy")[0] + "/manual/" + self.file.split(".")[0][-4:] + '.tif'):
            print("There's no original mask to compare! Analysis cannot be performed.")
            return
        self.nn_analysis = Analysis()
        epochs = [10, 25, 50, 75, 100]
        squared_errors = []
        for epoch in epochs:
            self.nn_train(validation_split=0.33, epochs=epoch)
            self.nn_predict()
            squared_error = self.nn_analysis.mean_squared_error(self.predicted_original, self.predicted)
            squared_errors.append([epoch, squared_error])
        print(squared_errors)
        file = open('squared_errors.txt', 'a')
        file.write(self.file)
        for error in squared_errors:
            file.write("epoch: " + str(error[0]) + "\t error: " + str(error[1]) + "\n")
        file.write("\n")
        file.close()

    def nn_tp(self):
        self.nn_train(validation_split=0.33, epochs=50)
        self.nn_predict()

    def browse(self):
        self.file = filedialog.askopenfilename()
        if len(self.file) > 0:
            self.load_images(self.file)
        if os.path.isfile(self.file.split("/healthy")[0] + "/manual/" + self.file.split(".")[0][-4:] + '.tif'):
            self.predicted_original = io.imread(self.file.split("/healthy")[0] + "/manual/" + self.file.split(".")[0][-4:] + '.tif')

    def display_picture(self, picture, picture_type):
        width = 500
        width_percent = width/float(picture.size[0])
        height = int((float(picture.size[0] * float(width_percent))))
        resized_picture = picture.resize((width, height), Image.ANTIALIAS)
        resized_picture = ImageTk.PhotoImage(resized_picture)
        label = Label(self, image=resized_picture)
        label.image = resized_picture
        if picture_type == 'input':
            label.place(x=30, y=50)
        elif picture_type == 'output':
            label.place(x=560, y=50)

    def load_images(self, file):
        self.input_picture = io.imread(file)
        self.display_picture(Image.fromarray(self.input_picture), 'input')
        self.image_processing = Image_processing.ImageProcessing()
        self.output_picture = self.image_processing.process_picture(self.input_picture)
        self.display_picture(Image.fromarray(self.output_picture), 'output')


if __name__ == '__main__':
    root = Tk()
    app = MainWindow(root, "pictures/images/01_hx.jpg")
    root.mainloop()
