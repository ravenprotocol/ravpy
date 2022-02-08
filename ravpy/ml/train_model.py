import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation,Dense
import os

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

def train_local(local_epochs=5):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    num_classes = 10
    num_training_images = 6000
    num_testing_images = 1000

    x_train = x_train[:num_training_images].reshape(num_training_images, 784)
    x_train = x_train / 255

    y_train = to_categorical(y_train[:num_training_images], num_classes)

    x_test = x_test[:num_testing_images].reshape(num_testing_images, 784)
    x_test = x_test / 255

    y_test = to_categorical(y_test[:num_testing_images], num_classes)

    batch_size = 32

    if len(os.listdir('model')) == 0:
        print("No Model Found. Initializing...")
        lr = 0.01
        total_comms_round = 100
        loss='categorical_crossentropy'
        metrics = ['accuracy']
        optimizer = SGD(learning_rate=lr,
                        decay=lr / total_comms_round,
                        momentum=0.9
                       )

        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss, 
                        optimizer=optimizer, 
                        metrics=metrics)

    else:    
        local_model = keras.models.load_model('model/local.h5')
        weights = local_model.get_weights()
        local_model.set_weights(weights)
            

    x = np.array(x_train)
    y = np.array(y_train)

    local_model.fit(x, y, epochs=local_epochs, verbose=1, batch_size=batch_size)

    score = local_model.evaluate(x_test, y_test, verbose=0)
    print("Test Score: ", score[1])

    local_model.save('model/local.h5')

train_local(5)
