import time
from data import get_training_data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

x_train, y_train = get_training_data()

# Normalise the data
x_train = x_train / 255.0

# Parameter options
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for num_dense_layers in dense_layers:
    for layer_size in layer_sizes:
        for num_conv_layers in conv_layers:
            NAME = f'{num_conv_layers}-conv-{layer_size}-nodes-{num_dense_layers}-dense-{int(time.time())}'
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

            # Create model
            model = Sequential()
            # (3, 3) is the convolution kernel size (window size)
            model.add(Conv2D(layer_size, (3, 3), input_shape = x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for _ in range(num_conv_layers - 1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            for _ in range(num_dense_layers):
                model.add(Dense(256))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            # Compile model
            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # Train model
            model.fit(x_train, y_train, batch_size=32, epochs=10,
                      validation_split=0.1, callbacks = [tensorboard])
