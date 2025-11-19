import tensorflow as tf
import keras
from keras import layers
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from data import generate_image_dataset
from sklearn.model_selection import train_test_split

X, y = generate_image_dataset()

RANDOM_SEED = 42

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