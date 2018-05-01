from skimage import io, color
import numpy as np
import skimage
import pandas as pd
import os


class Sample:
    input_path = 'pictures/images/'
    output_path = 'pictures/manual/'
    output_extension = '.tif'
    sample_path = 'samples/'


    @staticmethod
    def generate_csv(sample_size, sample_step, number_of_files, sample_path):
        file_list = [f for f in os.listdir(Sample.input_path) if f.split(".")[0][-1] == "h"]
        all_data_frames = pd.DataFrame(columns=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'result'])
        for file_name in file_list[:number_of_files]:
            data_frame = Sample.create_samples(file_name, sample_size=sample_size,
                                               step=sample_step, equals_set_sizes=True)
            all_data_frames = all_data_frames.append(data_frame)
        all_data_frames.to_csv(sample_path, index=False)
        print("File", sample_path, "is generated!")

    @staticmethod
    def create_samples(file_name, sample_size, step, equals_set_sizes):
        print(file_name)
        # sample_size has to be odd number
        assert sample_size % 2 == 1
        img = io.imread(Sample.input_path + file_name)
        name = file_name.split(".")
        output_img = io.imread(Sample.output_path + name[0] + Sample.output_extension)

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
                    samples_left -= 1

            print("Completion finished, in total", row, "images")

        return data_frame

