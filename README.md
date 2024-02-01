# Kernal_vs_Neural_Networks
Implementation and comparison between two kernel models and a neural network model in binary classification task on a partial dataset from MNIST using PyTorch.

# Introduction

### Kernel Methods:
Kernel methods are a class of algorithms used for pattern analysis, machine learning, and statistical inference. They are particularly useful when dealing with non-linearly separable data or when working with high-dimensional feature spaces. Kernel methods operate by implicitly mapping input data into a higher-dimensional space, where it becomes easier to separate or classify the data linearly. The key idea behind kernel methods is to use a function called a "kernel" to compute the inner products between pairs of data points in this higher-dimensional space without explicitly computing the transformations.

Two common types of kernels used in kernel methods are the Radial Basis Function (RBF) kernel and the logistic kernel:
#### Radial Basis Function (RBF) Kernel:
* The RBF kernel, also known as the Gaussian kernel, is one of the most widely used kernels in kernel methods.
* The RBF kernel assigns high similarity (or proximity) between points that are close to each other and lower similarity for points that are far apart.
* In classification tasks, the RBF kernel is particularly effective in capturing complex decision boundaries and handling non-linearly separable data.

#### Logistic Kernel:
* The logistic kernel is another type of kernel used in kernel methods, especially in support vector machines (SVMs) and other classification algorithms.
* The logistic kernel is used for transforming the input data into a higher-dimensional space where the classes become more separable.
* Similar to the RBF kernel, the logistic kernel allows for the modeling of non-linear decision boundaries and complex relationships in the data.

### Neural Networks:
Neural networks are arranged in layers: input, hidden (which extract features), and output. Through training, typically via backpropagation, the network adjusts the strengths of connections (weights) to minimize prediction errors. These models are adept at various tasks like classification, regression, and pattern recognition, finding extensive applications across domains like image and speech recognition, natural language processing, and autonomous systems.


In this assignment, I applied two kernel models and a neural network model to the binary classification task on a partial dataset from MNIST. In this classification task, the model takes a 16x16 image of handwritten digits as inputs and classify the image into two classes. For each data sample, the dictionary key x indicates its raw features, which are represented by a 256-dimensional vector where the values between [ 1; 1] indicate grayscale pixel values for a 16x16 image. In addition, the key y is the label for a data example, which can be 0, 1 or 2.

# Observation and Results

<img width="731" alt="image" src="https://github.com/ashuchauhan171996/Kernal_vs_Neural_Networks/assets/83955120/3880168f-de7c-4733-ba17-c552647da65c">

<img width="747" alt="image" src="https://github.com/ashuchauhan171996/Kernal_vs_Neural_Networks/assets/83955120/60fa972f-035a-4e0b-b9ae-ed27d06c95ad">

<img width="753" alt="image" src="https://github.com/ashuchauhan171996/Kernal_vs_Neural_Networks/assets/83955120/1a961536-b082-4203-a728-0b36ee3aaa1c">

# Summary
<img width="540" alt="image" src="https://github.com/ashuchauhan171996/Kernal_vs_Neural_Networks/assets/83955120/53126de3-8928-4342-baaa-565deaeda34f">





