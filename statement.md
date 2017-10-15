# Deep Learning From Scratch: Theory and Implementation

In this text, we develop the mathematical and algorithmic underpinnings of deep neural networks from scratch and implement our own neural network library in Python, mimicking the <a href="http://www.tensorflow.org">TensorFlow</a> API. I do not assume that you have any preknowledge about machine learning or neural networks. However, you should have some preknowledge of calculus, linear algebra, fundamental algorithms and probability theory on an undergraduate level. If you get stuck at some point, please leave a comment.

By the end of this text, you will have a deep understanding of the math behind neural networks and how deep learning libraries work under the hood.

I have tried to keep the code as simple and concise as possible, favoring conceptual clarity over efficiency. Since our API mimicks the TensorFlow API, you will know how to use TensorFlow once you have finished this text, and you will know how TensorFlow works under the hood conceptually (without all the overhead that comes with an omnipotent, maximally efficient machine learning API).

# Computational graphs
We shall start by defining the concept of a computational graph, since neural networks are a special form thereof. A computational graph is a directed graph where the nodes correspond to **operations** or **variables**. Variables can feed their value into operations, and operations can feed their output into other operations. This way, every node in the graph defines a function of the variables.

The values that are fed into the nodes and come out of the nodes are called <b>tensors</b>, which is just a fancy word for a multi-dimensional array. Hence, it subsumes scalars, vectors and matrices as well as tensors of a higher rank.

Let's look at an example. The following computational graph computes the sum $`z`$ of two inputs $`x`$ and $`y`$. 
Here, $`x`$ and $`y`$ are input nodes to $`z`$ and $`z`$ is a consumer of $`x`$ and $`y`$. $`z`$ therefore defines a function $`z : \mathbb{R^2} \rightarrow \mathbb{R}`$ where $`z(x, y) = x + y`$.

<img src="addition.png?456" style="height: 200px;">

The concept of a computational graph becomes more useful once the computations become more complex. For example, the following computational graph defines an affine transformation $`z(A, x, b) = Ax + b`$.

<img src="affine_transformation.png" style="height: 200px;">

## Operations

Every operation is characterized by three things:
- A `compute` function that computes the operation's output given values for the operation's inputs
- A list of `input_nodes` which can be variables or other operations
- A list of `consumers` that use the operation's output as their input

Let's put this into code:
