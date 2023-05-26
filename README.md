# OptimizationAlgorithms_EE_2023
This repository contains the code implementation for the optimization algorithms investigated in my Extended Essay (EE) as part of the International Baccalaureate (IB) Program. The code, written in Python, is designed to analyze and compare different optimization methods for solving a variety of mathematical functions.

The main focus of this project is to evaluate the performance and convergence speed of three optimization algorithms: Gradient Descent, Newton's Method, and BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm. The algorithms are tested on three distinct mathematical functions: Rosenbrock function, Sphere function, and Booth function.

The code includes functions to generate initial coordinates for each function, run multiple iterations of each algorithm, and measure the execution time for finding the global minimum. The data collected from these experiments is used to analyze and compare the effectiveness of the optimization algorithms.

This repository serves as a comprehensive resource for understanding the implementation and evaluation of optimization algorithms in the context of the Extended Essay. It provides a foundation for studying the convergence behavior of different methods and can be utilized as a reference for future research or experimentation.

To run the code in the provided repository, follow these steps:

1. Clone EEcode folder on your computer.

2. Start by compiling the Initial.py file. This will generate a text file named coordinates.txt containing a list of coordinates.

3. Once the coordinates.txt file is generated, you can run the Rosenbrock.py, Sphere.py, and Booth.py files. These files will perform the optimization algorithms on their respective functions and calculate the average compilation time for each algorithm.

4. Additionally, there is a design.py file that provides a visualization of the Newton's and BFGS algorithms on the Sphere function. However, note that the code for the Newton algorithm is currently commented out. If you want to visualize the Newton algorithm, uncomment that section and comment out the BFGS algorithm section.

By following these instructions, you should be able to execute the code and observe the results and visualizations.
