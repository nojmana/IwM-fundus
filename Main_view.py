from tkinter import *
from tkinter import filedialog
from tkinter.ttk import Frame, Button, Label
from PIL import Image, ImageTk
from skimage import io
from skimage.color import rgb2gray


class MainWindow(Frame):

    def __init__(self, root, file):
        super().__init__()
        self.root = root

        self.quit_button = Button(self, text="Quit", command=self.quit)
        self.browse_button = Button(self, text="Browse file", command=self.browse)
        self.init_ui()

        self.input_picture = io.imread(file)
        self.display_picture(Image.fromarray(self.input_picture), 'input')
        self.display_picture(Image.fromarray(self.input_picture), 'output')
        self.output_picture = None

    def init_ui(self):
        self.master.title("Detection of fundus blood vessels")
        self.pack(fill=BOTH, expand=1)
        self.center_window()

        self.quit_button.place(x=1200, y=20)
        self.browse_button.place(x=1100, y=20)

    def center_window(self):
        w = 1300
        h = 700

        x = (self.master.winfo_screenwidth() - w) / 2
        y = (self.master.winfo_screenheight() - h) / 2
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))

    def browse(self):
        file = filedialog.askopenfilename()
        if len(file) > 0:
            self.load_images(file)

    def display_picture(self, picture, picture_type):
        width = 500
        width_percent = width/float(picture.size[0])
        height = int((float(picture.size[0] * float(width_percent))))
        resized_picture = picture.resize((width, height), Image.ANTIALIAS)
        resized_picture = ImageTk.PhotoImage(resized_picture)
        label = Label(self, image=resized_picture)
        label.image = resized_picture
        if picture_type == 'input':
            label.place(x=100, y=100)
        elif picture_type == 'output':
            label.place(x=700, y=100)

    def load_images(self, file):
        self.input_picture = rgb2gray(io.imread(file))
        self.display_picture(Image.fromarray(self.input_picture), 'input')
        self.display_picture(Image.fromarray(self.input_picture), 'output')


if __name__ == '__main__':
    root = Tk()
    app = MainWindow(root, "pictures/healthy/01_h.jpg")
    root.mainloop()
