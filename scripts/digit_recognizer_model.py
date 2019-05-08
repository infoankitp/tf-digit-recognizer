from batch_utilities import BatchUtilities
from model_builder import ModelGenerator
import tensorflow as tf
import numpy as np
from sklearn import metrics
import traceback
import argparse
class DigitRecognizerModel():


    def __init__(self):
        self.model_saver_global_step = 10
        self.image_height = 28
        self.image_width = 28
        self.image_num_channels = 3
        self.num_classes = 10

        pass

    def fit(self, sess, input_image_shape, num_classes, learning_rate, num_epochs, batch_size, model_saving_location
            , model_name , model_saver_after_iterations, pretrained_model_object = None):
        height, width, num_channels = input_image_shape
        image_array, label_array, num_classes = None, None, num_classes
        batch_utils = BatchUtilities(image_array, label_array, num_classes)
        image_array, label_array, num_classes = batch_utils.image_array, batch_utils.label_array, batch_utils.num_labels

        # with tf.device("/gpu:0"):
        if pretrained_model_object is None:
            input_images = tf.placeholder(tf.float32, shape=[None, height, width, num_channels],
                                          name='input_images')
            input_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_labels')
            network = ModelGenerator().generate_model(input_images, num_classes)

        else:
            # {"network" : network, "input_images": input_images, "input_labels" : input_labels, "optimizer" : optimizer}git
            network = pretrained_model_object['network']
            input_images = pretrained_model_object['input_images']
            input_labels = pretrained_model_object['input_labels']
            optimizer = pretrained_model_object['optimizer']

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=network, labels=input_labels)
        cost = tf.reduce_mean(cross_entropy)
        tf.summary.scalar("cost", cost)
        merged = tf.summary.merge_all()

        if pretrained_model_object is None:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(model_saving_location + model_name, graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)
        loss_array = []

        counter = 0
        for epoch in range(num_epochs):
            batch_utils = BatchUtilities(image_array, label_array, num_classes)
            print("Doing Epoch : ", epoch + 1)
            images, labels = batch_utils.get_random_batch(batch_size)
            while images is not None or labels is not None:
                counter += 1
                loss, summary = sess.run([cost, merged], feed_dict={input_images: images, input_labels: labels})
                sess.run(optimizer, feed_dict={input_images: images, input_labels: labels})
                images, labels = batch_utils.get_random_batch(batch_size)
                writer.add_summary(summary, counter)
            if epoch%model_saver_after_iterations == 0:
                saver.save(sess, model_saving_location + model_name, global_step=epoch)
            loss_array.append(loss)
            if epoch % 10 == 0:
                print('loss', loss)
        return loss_array


    def restore_pretrained_model(self, sess, model_saved_location, model_name, model_saver_after_iterations):
        try:
            saver = tf.train.import_meta_graph('{}{}-{}.meta'.format(model_saved_location, model_name, model_saver_after_iterations))
            saver.restore(sess, tf.train.latest_checkpoint(model_saved_location))  # search for checkpoint file
            graph = tf.get_default_graph()
            network = graph.get_tensor_by_name("add_4:0")
            input_labels = graph.get_tensor_by_name("input_labels:0")
            input_images = graph.get_tensor_by_name("input_images:0")
            optimizer = graph.get_operation_by_name("Adam")
            pretrained_model_object = {"network": network, "input_images": input_images, "input_labels": input_labels,
                                       "optimizer": optimizer}
            return pretrained_model_object
        except Exception as err:
            print("Error Restoring Pretrained Model")
            print(err)
            print(traceback.format_exc())
        return None

    def train(self,  model_saved_location, model_name , model_saver_after_iterations, learning_rate, num_epochs, batch_size):
        tf.reset_default_graph()
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        pretrained_model_object = self.restore_pretrained_model(sess, model_saved_location, model_name, model_saver_after_iterations)
        if pretrained_model_object is None:
            loss_array = self.fit(sess, [self.image_height, self.image_width, self.image_num_channels],
                              self.num_classes, batch_size=batch_size, num_epochs = num_epochs, learning_rate = learning_rate,
                                  model_saving_location = model_saved_location, model_name = model_name, model_saver_after_iterations = model_saver_after_iterations)
        else:
            loss_array = self.fit(sess, [self.image_height, self.image_width, self.image_num_channels],
                                  self.num_classes, batch_size=batch_size,pretrained_model_object=pretrained_model_object,
                                  num_epochs = num_epochs, learning_rate = learning_rate, model_saving_location =
                                  model_saved_location, model_name = model_name, model_saver_after_iterations = model_saver_after_iterations)
        return loss_array


    def predict(self, model_saved_location, model_name, model_saver_after_iterations, images):
        tf.reset_default_graph()
        config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        pretrained_model_object = self.restore_pretrained_model(sess, model_saved_location, model_name, model_saver_after_iterations)
        if pretrained_model_object is not None:

            self.pretrained_model_object = pretrained_model_object
            network = pretrained_model_object['network'] # Place Holder of Last tensor in the network
            network = tf.nn.softmax(network) # Adding a Softmax to the existing model
            input_images = pretrained_model_object['input_images']
            result = sess.run(network, feed_dict={input_images: images})
            predictions = np.argmax(result, axis=1)
            scores = np.amax(result, axis=1)
            return {'predictions' : predictions, "scores" : scores}
        else:
            print("Could Not load the Model")
        return None


    def get_accuracy(self,model_saved_location, model_name, model_saver_after_iterations, data_location, batch_size):
        batch_util = BatchUtilities(None, None, None, data_location)
        images, labels = batch_util.get_random_batch(batch_size)
        true_labels = []
        score_array = []
        prediction = []
        while images is not None or labels is not None:
            rslt = self.predict(model_saved_location, model_name, model_saver_after_iterations, images)
            if rslt is not None:
                tmp_true_labels = np.argmax(labels, axis=1)
                tmp_prediction = rslt['predictions']
                tmp_score_array = rslt['scores']
                true_labels.extend(tmp_true_labels)
                prediction.extend(tmp_prediction)
                score_array.extend(tmp_score_array)
            accuracy = metrics.accuracy_score(true_labels, prediction)
            print("Accuracy of the Model on the Test Data : ",accuracy*100)
            return {'prediction' : prediction, 'true_labels' : true_labels, 'score_array' :score_array,
                    'accuracy' : accuracy}
        else:
            return None


def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest = "action", help = 'Train : If model is to be trained\ncheck_accuracy : if accuracy is to be chekced on a test data.')
    subparser_train = subparser.add_parser('train')

    subparser_train.add_argument('-a', '--learning-rate', help="Learning Rate", default=0.01, type = float)
    subparser_train.add_argument('-n', '--num-epochs', help="Number of Epochs to be trained with", default=100, type = int)


    subparser_accuracy = subparser.add_parser('check_accuracy')
    subparser_accuracy.add_argument("--data-location", help = "Location at which test data is located", required= True)



    parser.add_argument('--model-saved-location', help="System Path at which model is to be saved", default='../model/')
    parser.add_argument('--model-name', help="Name of the model to be saved with", default='digit_recognizer')
    parser.add_argument('--model-saver-after-iterations', help="Number of iterations after which model is to be saved", default=10, type = int)
    parser.add_argument('-s', '--batch-size', help="Number of images to be processed in a batch", default=1000, type = int)
    args = parser.parse_args()
    if args.action == 'train':
        model_obj = DigitRecognizerModel()
        model_obj.train(model_saved_location=args.model_saved_location, model_name=args.model_name,
                        model_saver_after_iterations= args.model_saver_after_iterations, batch_size= args.batch_size,
                        learning_rate=args.learning_rate, num_epochs=args.num_epochs)
    elif args.action == 'check_accuracy':
        model_obj = DigitRecognizerModel()
        model_obj.get_accuracy(model_saved_location=args.model_saved_location, model_name=args.model_name,
                               model_saver_after_iterations= args.model_saver_after_iterations,
                               batch_size= args.batch_size, data_location = args.data_location)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()