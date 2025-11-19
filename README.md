# Neural Network - Image Classifier

Building a neural network to classify images - training the neural network using a generated dataset of images - evaluating the performance of the neural network

- The requirements for this repo can be found in **requirements.txt**

- The training dataset is generated in **data.py**. Each 'image' is an 8x8 NumPy array, and belongs to one of these types: "horizontal line", "vertical line", "blank". A blank image consists of only 0s, whereas horizontal and vertical lines are 1s.

- A training dataset can be created by importing the generate_image_dataset function from data.py: X, y = generate_image_dataset()

- The neural network is built, trained, and evaluated in **main.py**. It is fully connected and has one input layer, two hidden layers, and an output layer. The helper function print_image prints predicted images to the console and shows what the model predicted vs. what it really is. 
