import tensorflow
from data import generate_image_dataset
from sklearn.model_selection import train_test_split

X, y = generate_image_dataset()

RANDOM_STATE_SEED = 24601

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE_SEED)

