import pandas as pd
import cv2
import numpy as np
from scripts.digit_recognizer_model import DigitRecognizerModel


class HandWrittenDigitPredictor():
    def __init__(self):
        self.model_saved_location = "../model/"
        self.model_name = "digit_recognizer"
        self.model_saver_global_step = 10
        pass

    def convert_array_into_images(self, img_array):
        img = np.array(img_array).reshape(28, 28)
        img = img.astype('uint8')

        backtorgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return backtorgb


    def process_data(self, csv_file_location, output_file_location, batch_size = 1000):

        test_data = pd.read_csv(csv_file_location)
        test_images = []
        for index, row in test_data.iterrows():
            img = self.convert_array_into_images(row['pixel0':])
            test_images.append(img)
        total_num_images = len(test_images)
        current_index = 0
        result_array = []
        prediction_array = []
        model_obj = DigitRecognizerModel()
        batch_number = 1
        while current_index < total_num_images :
            print("Processing Batch Number : ", batch_number)
            result = model_obj.predict(self.model_saved_location, self.model_name, self.model_saver_global_step, test_images[current_index : current_index + batch_size])
            prediction_array.extend(result['predictions'])
            current_index += batch_size
            batch_number += 1

        rslt_df = pd.DataFrame(prediction_array)
        rslt_df.index += 1
        rslt_df.index.names = ['ImageId']
        rslt_df.columns = ['Label']
        rslt_df.to_csv(output_file_location, header = True)
        print("Wrote Data File : ", output_file_location)


if __name__ == '__main__':
    obj = HandWrittenDigitPredictor()
    obj.process_data("../data/test.csv", "../data/output.csv")


