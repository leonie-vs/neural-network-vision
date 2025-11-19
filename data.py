import numpy as np

def generate_blank():
    return np.zeros((8,8))
    
def generate_horizontal_line():
    random_index = np.random.randint(0,8)
    blank_array = generate_blank()
    blank_array[random_index] = 1
    return blank_array

print(generate_horizontal_line())
