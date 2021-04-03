# CS464 Introduction to Machine Learning
The assignment and project implementations for CS464 course, Bilkent University.

## Coverage of the repository

### HW1
Probability Review and Naive Bayes
- Apply independence onto conditional probabilities to solve a probability problem. 
- Derive MLE and MAP estimates for a Poisson distribution and prove they are the same under a uniform prior.
- On Twitter Airline Dataset, use Multinomial Naive Bayes to apple sentiment classification with a Bag-of-Words approach. 
- On the same data, apply Binomial Naive Bayes models. 
- Compare and discuss the results. 
- Implemented in Python (Jupyter Notebook)

### HW2
PCA, Linear Regression, Logistic Regression and SVM (Support Vector Machine)
- Implement and use PCA to obtain eigenfaces and recionstruct the images with a certain number of PCs. Discuss the results. 
- Derive the closed form solution for the ordinary least squares (OLS) loss for linear regression, and apply the solution to the presented data. 
- Apply full-batch, and mini-batch gradient descent algorithm to logistic regression classifier, display a manually constructed confusion matrix and performance metrics. 
- Apply an SVM to the same dat using sklearn library. Discuss the results. 

### HW3
Multilayered Perceptrons and Convolutional Neural Networks 
- **This homework has been implemented using PyTorch.**
- Create a DataLoad to load the data and implement a custom dataset class.
- Train on the data using both Transfer Learning with ResNet-18 and from scratch. Create test statistics and compare the results. 
- Use DataLoader to load the fine-grained bird classification dataset. 
- Design MLP and CNN to train the networks. Visualize and comment on the feature maps. 
- Implemented in Python (Jupyter Notebook)

###  Project:
Steering Wheel Angle Regression using Deep Learning 
- Sampled driving frames and the corresponding angles from the ROS data.
- Used and optimized,
	-  Transfer Learning, 
	- MLP and 
	- CNN architectures. 


