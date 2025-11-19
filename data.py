import numpy as np

def generate_blank():
    return np.zeros((8,8))
    
def generate_horizontal_line():
    random_index = np.random.randint(0,8)
    blank_array = generate_blank()
    blank_array[random_index] = 1
    return blank_array

def generate_vertical_line():
    random_index = np.random.randint(0,8)
    blank_array = generate_blank()
    for line in blank_array:
        line[random_index] = 1
    return blank_array

def generate_image_dataset(samples_per_class=10):

    data = []
    labels = []

    for i in range(samples_per_class):
        result = generate_blank()
        result.flatten()
        data.append(result)
        labels.append(0)

    for i in range(samples_per_class):
        result = generate_horizontal_line()
        result.flatten()
        data.append(result)
        labels.append(1)
    
    for i in range(samples_per_class):
        result = generate_vertical_line()
        result.flatten()
        data.append(result)
        labels.append(2)

    X = np.array(data)
    y = np.array(labels)

    return X, y

X, y = generate_image_dataset()
print(X, y)