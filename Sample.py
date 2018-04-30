from skimage import io, color
import numpy as np
import skimage
import pandas as pd


class Sample:
    input_path = 'pictures/images/'
    output_path = 'pictures/manual/'
    output_extension = '.tif'
    sample_path = 'samples/'

    def create_samples2(file_name, sample_size, step, equals_set_sizes):
        print(file_name)
        # sample_size has to be odd number
        assert sample_size % 2 == 1
        img = io.imread(Sample.input_path + file_name)
        name = file_name.split(".")
        output_img = io.imread(Sample.output_path + name[0] + Sample.output_extension)

        pic_counter = 1
        nerve_counter = 0
        not_nerve_counter = 0
        number_of_pics = int((img.shape[0] - sample_size) / step) * int((img.shape[1] - sample_size) / step)

        if equals_set_sizes:
            print("Creating samples is in progress! This method will create approximately",
                  int(1.9 * number_of_pics), "pictures")
        else:
            print("Creating samples is in progress! This method will create", number_of_pics, "pictures")

        data_frame = pd.DataFrame(columns=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'result'])

        row = 0
        img = skimage.color.rgb2gray(img)
        for i in np.arange(0, img.shape[0] - sample_size, step):
            for j in np.arange(0, img.shape[1] - sample_size, step):
                if output_img[int(i + (sample_size / 2)), int(j + (sample_size / 2))] == 0:
                    result = 0
                    not_nerve_counter += 1
                else:
                    result = 1
                    nerve_counter += 1

                current_hu = skimage.measure.moments_hu(img[i:i + sample_size, j:j + sample_size])
                current_hu = np.append(current_hu, result)
                data_frame.loc[row] = current_hu
                row += 1
                pic_counter += 1
        print("All samples are created! Y:", nerve_counter, "N:", not_nerve_counter)

        if equals_set_sizes:
            samples_left = not_nerve_counter - nerve_counter
            print("Completion started! Images to generate:", samples_left)
            while samples_left > 0:
                rand_x = np.random.randint(sample_size, img.shape[0] - sample_size)
                rand_y = np.random.randint(sample_size, img.shape[1] - sample_size)
                if output_img[rand_x][rand_y] != 0:
                    result = 1
                    current_hu = skimage.measure.moments_hu(img[rand_x:rand_x + sample_size, rand_y:rand_y + sample_size])
                    current_hu = np.append(current_hu, result)
                    data_frame.loc[row] = current_hu
                    row += 1
                    pic_counter += 1
                    samples_left -= 1

            print("Completion finished, in total", pic_counter - 1, "images")

        return data_frame

