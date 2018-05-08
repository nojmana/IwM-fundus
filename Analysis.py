import numpy as np


class Analysis:
    file_name = 'confusion.txt'

    @staticmethod
    def mean_squared_error(original, output):
        return np.square(np.subtract(original, output)).mean()

    @staticmethod
    def confusion(predicted, original):
        assert predicted.shape == original.shape
        size = predicted.shape[0] * predicted.shape[1]
        true_positive = true_negative = false_positive = false_negative = 0
        for i in range(len(predicted)):
            for j in range(len(predicted[i])):
                if predicted[i,j] == 0 and original[i,j] == 0:
                    true_positive += 1
                elif predicted[i,j] != 0 and original[i,j] != 0:
                    true_negative += 1
                elif predicted[i,j] == 0 and original[i,j] != 0:
                    false_positive += 1
                else:
                    false_negative += 1
        file = open(Analysis.file_name, 'a')
        file.write("\n")
        file.write("tp " + str(true_positive) + " tn " + str(true_negative) + " fp " + str(false_positive) + " fn " +
                   str(false_negative) + " size " + str(size) + "\n")
        file.write(str(round(true_positive*100/size, 2)) + " " + str(round(true_negative*100/size, 2)) + " " +
                   str(round(false_positive*100/size, 2)) + " " + str(round(false_negative*100/size, 2)) + " \n")
        file.write("\n")
        file.close()
