# Convolutional Neural Networks (CNN) - Deep Learning Specialization

This repository contains the core concepts, architectures, and implementations from the Convolutional Neural Networks course. The focus is on applying deep learning to computer vision tasks, including image classification, object detection, and neural style transfer.

## 🛠 Core Components

### 1. Convolution Operation
The building block of CNNs. It uses filters (kernels) to detect features such as edges, textures, and complex shapes.
* **Padding:** Adding borders of zeros to the input matrix to allow for "Same" convolution (output size = input size).
* **Stride:** The number of pixels the filter skips as it slides over the image.
* **Channels:** Convolution operates across the depth of the input (e.g., RGB).

### 2. Pooling Layers
Reduces the spatial dimensions (width and height) of the representation to decrease the number of parameters and computational load.
* **Max Pooling:** Picks the maximum value from the window.
* **Average Pooling:** Computes the average value of the window.

### 3. Fully Connected (FC) Layers
Standard neural network layers used at the end of the network to flatten the volume and perform final classification.

---

## 🏗 Classic Architectures

| Architecture | Key Innovation |
| :--- | :--- |
| **LeNet-5** | Introduced the basic Conv-Pool-Conv-Pool-FC structure. |
| **AlexNet** | Used ReLU and Dropout; significantly deeper than LeNet. |
| **VGG-16** | Focused on simplicity, using only $3 \times 3$ convolutions and $2 \times 2$ pooling. |
| **ResNet** | Introduced **Residual Blocks** (skip connections) to solve the vanishing gradient problem in very deep networks. |
| **Inception (GoogLeNet)** | Uses $1 \times 1$ convolutions to reduce dimensionality and concatenates multiple filter sizes at the same level. |

---

## 🔍 Advanced Applications

### Object Detection (YOLO)
"You Only Look Once" (YOLO) is a state-of-the-art, real-time object detection algorithm. It treats detection as a regression problem to spatially separated bounding boxes and associated class probabilities.
* **Anchor Boxes:** Allow the model to detect multiple objects in a single grid cell.
* **Non-Max Suppression (NMS):** Removes overlapping bounding boxes that point to the same object.
* **Intersection over Union (IoU):** Evaluates the overlap between the predicted and ground truth boxes.

### Face Recognition
* **One-Shot Learning:** Learning to recognize a person from just one example.
* **Siamese Networks:** Training a network to output an encoding (embedding) where the distance between two images of the same person is minimized.
* **Triplet Loss:** A loss function that compares an Anchor, a Positive, and a Negative image.

### Neural Style Transfer
Combining the **content** of one image with the **style** of another.
* **Content Cost:** Calculated using the activations of a hidden layer.
* **Style Cost:** Calculated using the **Gram Matrix** to capture correlations between features across channels.

---

## 🚀 Implementation Summary

The implementations in this specialization are typically built using **TensorFlow/Keras**. 

### Example CNN Structure (Keras)
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(units=classes, activation='softmax')
])
