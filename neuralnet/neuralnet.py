from tensorflow import keras as ks
import tensorflow as tf
import numpy as np

class NeuralNet:
    def __init__(
        self,
        neural_network_dimensions=(10),
        board_size=3,
        learning_rate=0.01,
        activation_functions=["sigmoid", "tanh"],
        optimizer_name="adam",
        load_saved_model=False,
        episode_number=0,
    ):
        self.board_size = board_size
        if load_saved_model:
            try:
                self.model = self.load_saved_model(episode_number)
            except OSError:
                raise ValueError(
                    f"Failed to load model named {self.board_size}x{self.board_size}_ep{episode_number}. Did you provide the correct episode number?"
                )
        else:
            self.model = self.init_model(
                neural_network_dimensions, board_size, learning_rate, activation_functions, optimizer_name
            )
        self.topp = load_saved_model


    def init_model(self, neural_network_dimensions, board_size, learning_rate, activation_functions, optimizer_name):
        # Get activation functions for actor and critic
        actor_activation_function = activation_function.get(activation_functions[0])
        critic_activation_function = activation_function.get(activation_functions[1])
        
        # Check if valid activation functions were provided
        if not actor_activation_function or not critic_activation_function:
            raise ValueError(f"Invalid activation function provided. Must be one of: {', '.join(activation_function.keys())}")
        
        # Create input layer
        input_layer = ks.layers.Input(shape=(board_size ** 2 + 1))
        
        # Create actor and critic layers
        actor_layer = ks.layers.Dense(board_size ** 2 + 1, activation=actor_activation_function)(input_layer)
        critic_layer = ks.layers.Dense(board_size ** 2 + 1, activation=critic_activation_function)(input_layer)
        
        # Add additional layers according to neural_network_dimensions
        for dim in neural_network_dimensions:
            actor_layer = ks.layers.Dense(dim, activation=actor_activation_function)(actor_layer)
            critic_layer = ks.layers.Dense(dim, activation=critic_activation_function)(critic_layer)
        
        # Create actor and critic output layers
        actor_output = ks.layers.Dense(board_size ** 2)(actor_layer)
        actor_output = ks.layers.Activation(activation="softmax", name="actor_output")(actor_output)
        
        critic_output = ks.layers.Dense(1)(critic_layer)
        critic_output = ks.layers.Activation(activation=critic_activation_function, name="critic_output")(critic_output)
        
        # Get optimizer
        optimizer = optimizers.get(optimizer_name)
        
        # Check if valid optimizer was provided
        if not optimizer:
            raise ValueError(f"Invalid optimizer provided. Must be one of: {', '.join(optimizers.keys())}")
        
        # Compile model
        model = ks.Model(inputs=input_layer, outputs=[actor_output, critic_output])
        
        losses = {"actor_output": "kl_divergence", "critic_output": "hinge"}
        loss_weights = {"actor_output": 1.0, "critic_output": 0.0}
        
        model.compile(optimizer=optimizer(learning_rate=learning_rate), loss=losses, loss_weights=loss_weights)
        
        model.summary()
        
        return model

    def fit(self, batch):
        if self.topp:
            raise Exception("Cannot train model when using TOP")
        input_data = []
        for state, _, _ in batch:
            state_data = []
            for i in state.split():
                state_data.append(int(i))
            input_data.append(state_data)
        input_data = np.array(input_data)
        actor_target_data = []
        for _, D, _ in batch:
            actor_target_value = D[0][1]
            actor_target_data.append(actor_target_value)
        actor_target_data = np.array(actor_target_data)
        ccritic_target_data = []
        for _, _, Q in batch:
            critic_target_value = Q
            critic_target_data.append(critic_target_value)
        critic_target_data = np.array(critic_target_data)
        target_data = {"actor_output": actor_target_data,
                    "critic_output": critic_target_data}
        self.model.fit(input_data, target_data, verbose=1, batch_size=64)

    def predict(self, input_data):
        model_predictions = self.model(input_data)
        prediction_length = len(model_predictions[0])
        legal_moves = []
        for n in range(prediction_length):
            normalized_model_predictions = []
            for i in range(len(model_predictions[0][n])):
                if input_data[0][i+1] == 0:
                    normalized_model_predictions.append(model_predictions[0][n][i])
                else:
                    normalized_model_predictions.append(0)
            normalized_model_predictions = np.array(normalized_model_predictions)
            normalized_model_predictions = NeuralNet.normalize(normalized_model_predictions)
            legal_moves.append(normalized_model_predictions)
        legal_moves = np.array(legal_moves)
        return legal_moves, model_predictions[1]
    

    def best_action(self, normalized_predictions):
        best_index = np.argmax(normalized_predictions[0])
        best_move = NeuralNet.convert_to_2d_move(best_index, self.board_size)
        return best_move

    def save_model(self, model_name, episode_number):
        model_path = "project2/models/%s%s.h5" % (model_name, episode_number)
        self.model.save(model_path)
        print("%s%s saved" % (model_name, episode_number))

    def load_saved_model(self, episode_number):
        model_path = "project2/models/%sx%s_ep%s.h5" % (self.board_size, self.board_size, episode_number)
        loaded_model = ks.models.load_model(model_path, compile=False)
        print("%sx%s_ep%s loaded" % (self.board_size, self.board_size, episode_number))
        return loaded_model

    @staticmethod
    def convert_to_2d_move(index, board_size):
        row = index // board_size
        col = index % board_size
        return (row, col)

    @staticmethod
    def normalize(array):
        # Assumes input of 1d np-array
        array_sum = sum(array)
        if array_sum == 0:
            return array
        return array / array_sum

    def safelog(tensor, base=0.0001):
        safe_tensor = tf.math.maximum(tensor, base)
        return tf.math.log(safe_tensor)

    def deepnet_cross_entropy(targets, outputs):
        log_softmax_outputs = tf.nn.log_softmax(outputs)
        cross_entropy = -1 * targets * log_softmax_outputs
        cross_entropy_sum = tf.reduce_sum(cross_entropy, axis=[1])
        mean_cross_entropy = tf.reduce_mean(cross_entropy_sum)
        return mean_cross_entropy
    
# Static values used to select activation function
activation_function = {
    "linear": ks.activations.linear,
    "sigmoid": ks.activations.sigmoid,
    "tanh": ks.activations.tanh,
    "relu": ks.activations.relu,
}

# Static values used to select optimizer
optimizers = {
    "adagrad": ks.optimizers.Adagrad,
    "sgd": ks.optimizers.SGD,
    "rmsprop": ks.optimizers.RMSprop,
    "adam": ks.optimizers.Adam,
}