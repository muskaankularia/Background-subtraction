# Background-subtraction
Background subtraction for videos

It was implemented using mixture of Gaussian model given in ”Adaptive background mixture models for real-time tracking” proposed by Strauffer and Grimson. It is done by fitting k-gaussians on each pixel, which model the pixel value distribution using E.M. algorithm.The results were compared with OpenCV library.

The program has these hyperparameters - 
k - number of gaussians to model each pixel
alpha - learning rate of algorithm
T - decides no. of gaussians representing background
