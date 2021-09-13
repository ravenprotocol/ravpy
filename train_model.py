import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

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
