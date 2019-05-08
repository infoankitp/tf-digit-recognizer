import os
import random
import cv2

class BatchUtilities():
    def __init__(self, image_array, label_array, num_labels, data_location = '../data/images/'):
        self.batch_image_counter = 0
        if image_array is None:
            self.image_array, self.label_array, self.num_labels = self.get_all_data(data_location)
        else:
            self.image_array, self.label_array, self.num_labels = image_array, label_array, num_labels
        self.total_input_images_size = len(self.image_array)

    def shuffle_image_label_list(self, image_array, label_array):
        tmp = list(zip(image_array, label_array))
        random.shuffle(tmp)
        image_array, label_array = zip(*tmp)
        return image_array, label_array

    def get_all_data(self, data_dir):
        image_array = []
        label_array = []
        directories = next(os.walk(data_dir))[1]
        labels = [x[0] for x in directories]
        num_labels = len(labels)
        for label in labels:
            label_folder_path = os.path.join(data_dir, label)
            files = os.listdir(label_folder_path)
            print("Loading Images!!!!!")
            for file in files:
                img = cv2.imread(os.path.join(label_folder_path, file))
                image_array.append(img)
                one_hot_label = [0] * num_labels
                one_hot_label[int(label)] = 1
                label_array.append(one_hot_label)
        image_array, label_array = self.shuffle_image_label_list(image_array, label_array)
        print("Read All Data!")
        return image_array, label_array, num_labels

    def get_random_batch(self, batch_size=200):

        if self.batch_image_counter >= self.total_input_images_size:
            data = None, None
        elif self.batch_image_counter + batch_size > self.total_input_images_size:
            data = self.image_array[self.batch_image_counter: self.batch_image_counter + batch_size], self.label_array[
                                                                                                      self.batch_image_counter: self.total_input_images_size]
        data = self.image_array[self.batch_image_counter: self.batch_image_counter + batch_size], self.label_array[
                                                                                                  self.batch_image_counter: self.batch_image_counter + batch_size]
        self.batch_image_counter = self.batch_image_counter + batch_size
        if len(data[0]) == 0:
            return None, None
        return data
