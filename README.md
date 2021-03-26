# Getting Started with CUDA Accelerated OpenCV

> This repository contains the code presented in the GTC2021 S31701
  talk.

## Description

This project presents a series of programs that guide you through the
process of optimizing a CUDA accelerated OpenCV algorithm. This
optimization is done through a series of well defined steps without
getting into low-level CUDA programming.

The algorithm chosen to illustrate the optimization process is the
calculation of the [magnitude of the Sobel
Derivatives](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html). While
not very interesting on its own, this algorithm is a foundational step
in many algorithms such as edge detection, image segmentation, feature
extraction, computer vision and more. While many optimizations can be
achieved by approximating the underlying math, the original definition
is kept for didactic purposes. The purpose is to focus the study on
the appropriate OpenCV+CUDA handling.

<p align="center">
  <img src="dog.jpg" alt="Original image of a cute, big-eyed puppy in grayscale" title="Original image of Bartok the Dachshund" width="300"/>
  <img src="dog_gradient.jpg" alt="Resulting gradient image" title="Gradient image of Bartok the Dachshund" width="300"/>
</p>

## Building the project

As usual with OpenCV projects, the chosen build system was
CMake. Start by making sure you have these dependencies installed:
* CMake
* OpenCV (with CUDA enabled)

Then proceed normally as follows:
```bash
# Clone the project
git clone https://github.com/RidgeRun/getting-started-with-cuda-opencv.git
cd getting-started-with-cuda-opencv

# Configure the project
mkdir build
cd build
cmake ..

# Build the project
make
```

If everything went okay, you should be able to run the demos. You may
specify the input and output images as the first and second parameters
respectively. Otherwise, "dog.jpg" and "dog_gradient_XXX.jpg" will be
used by default.

```bash
# Run from the build directory
./sobel_cpu ../dog.jpg

# Specify an alternative output
./sobel_cpu ../dog.jpg alternative_output.jpg

# Run from top-level with default parameters
cd ..
./build/sobel_cpu
```
