from model_tools import ModelTools
class ModelGenerator():

    def generate_model(self, input_images, num_labels):
        model = ModelTools()
        # Convolution Layer -1
        network = model.conv_layer(input_images, kernel=3, input_channels=3, output_channels=16, stride_size=1)
        network = model.pool_layer(network, kernel=2, stride=2)
        network = model.activation_layer(network)
        #         print(network)

        network = model.conv_layer(network, kernel=5, input_channels=16, output_channels=32, stride_size=1)
        network = model.pool_layer(network, kernel=2, stride=2)
        network = model.activation_layer(network)
        #         print(network)

        network = model.conv_layer(network, kernel=3, input_channels=32, output_channels=64, stride_size=1)
        network = model.pool_layer(network, kernel=3, stride=1)
        network = model.activation_layer(network)
        #         print(network)

        network, layer_size = model.flattening_layer(network)
        network = model.fully_connected_layer(network, layer_size, 512)
        network = model.activation_layer(network)

        network = model.fully_connected_layer(network, 512, num_labels)

        return network

