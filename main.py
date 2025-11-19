import random
import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from data import generate_image_dataset
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
os.environ["PYTHONHASHSEED"] = f"{RANDOM_SEED}"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

X, y = generate_image_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

OUTPUT_EVERY_X_EPOCHS = 2
print_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: 
    print(f"Epoch {epoch}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}") 
    if epoch % OUTPUT_EVERY_X_EPOCHS == 0 else None
)

# define Sequential model with four layers
model = Sequential([
    Input(shape=(64,)), # input layer
    Dense(32, activation='relu'), # first hidden layer
    Dense(16, activation='relu'), # second hidden layer
    Dense(3, activation='softmax') # output layer
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# train the model
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# evaluate the trained model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)

# helper function to print an 8×8 image in the console
def print_image(flat_array):
    img = flat_array.reshape(8, 8)
    for row in img:
        line = ''.join('#' if pixel == 1 else ' ' for pixel in row)
        print(line)

# mapping class numbers to labels
label_names = {
    0: "blank",
    1: "horizontal line",
    2: "vertical line"
}

# check five random predictions
pred_probs = model.predict(X_test)
pred_classes = np.argmax(pred_probs, axis=1)

indices = np.random.choice(len(X_test), 5, replace=False)

# print images to the console along with what the model thinks they are and what they really are
for i in indices:
    print("\nIMAGE:")
    print_image(X_test[i])

    true_label = y_test[i]
    pred_label = pred_classes[i]

    print("True label:     ", label_names[true_label])
    print("Predicted label:", label_names[pred_label])