deep-learning-challenge
Overview
This project aims to develop a binary classifier that can predict the likelihood of applicants achieving success if they receive funding from Alphabet Soup. The project will utilize the features present in the given dataset and employ diverse machine learning methods to train and assess the model's performance. The objective is to optimize the model in order to attain an accuracy score surpassing 75%.

Results
Data Preprocessing
The model aims to predict the success of applicants if they receive funding. This is indicated by the IS_SUCCESSFUL column in the dataset which is the target variable of the model. The feature variables are every column other than the target variable and the non-relevant variables such as EIN and names. The features capture relevant information about the data and can be used in predicting the target variables, the non-relevant variables that are neither targets nor features will be drop from the dataset to avoid potential noise that might confuse the model.
During preprocessing, I implemented binning/bucketing for rare occurrences in the APPLICATION_TYPE and CLASSIFICATION columns. Subsequently, I transformed categorical data into numeric data using the one-hot encoding technique. I split the data into separate sets for features and targets, as well as for training and testing. Lastly, I scaled the data to ensure uniformity in the data distribution.

Compiling, Training, and Evaluating the Model
Initial Model: For my initial model, I decided to include 3 layers: an input layer with 80 neurons, a second layer with 30 neurons, and an output layer with 1 neuron. I made this choice to ensure that the total number of neurons in the model was between 2-3 times the number of input features. In this case, there were 43 input features remaining after removing 2 irrelevant ones. I selected the relu activation function for the first and second layers, and the sigmoid activation function for the output layer since the goal was binary classification. To start, I trained the model for 100 epochs and achieved an accuracy score of approximately 74% for the training data and 72.9% for the testing data. There was no apparent indication of overfitting or underfitting.

AlphabetSoupCharity:
I attempted to optimize the model’s performance by first modified the architecture of the model by adding 2 dropout layers with a rate of 0.5 to enhance generalization and changed the activation function to tanh in the input and hidden layer. With that I got an accuracy score of 74.1% for my training set and 72.9% for my testing set.

Optimization attempts
1. AlphabetSoupCharity_Optimization:
for my first optimization, I tried to optimize the model’s performance by first modified the architecture of the model by adding 2 dropout layers with a rate of 0.5 to enhance generalization and changed the activation function to tanh in the input and hidden layer. Added one more layer and 150 Epochs to see if the model perfomance gets changed. With that I got an accuracy score of 74.1% for my training set and 72.6% for my testing set.

2. AlphabetSoupCharity_Optimization_1:
For my second optimization attempt, I used hyperparameter tuning. During this process, Keras identified the optimal hyperparameters, which includes using the tanh activation function, setting 32 neurons for the first layer, and assigning 4, 11, and 26 units for the subsequent layers. As a result, the model achieved an accuracy score of 73.3%.

3. AlphabetSoupCharity_Optimization_2:
For my second optimization attempt, I used three hiddern layers with tanh activation function and for the output layer I used sigmoid activation function. I tried changing the epoch to 20 as well.Still, It did not change the accuracy of the model. It is 73%. So, changing the sizes of hidden layers, neurons, epochs did not help much.

4. AlphabetSoupCharity_Optimization_3:
In my final optimization attempt, I removed the STATUS column and ASK AMT column in the preprocessing phase. Used relu activation function with two hidden layer and sigmoid function for the output layer with epochs of 100 that also did not change the performance pf the machine. I still get the Accuracy of 72.8%.

Summary
Given that I couldn't attain the target accuracy of 75%, I wouldn't suggest any of the models above. However, with additional time, I would explore alternative approaches like incorporating the Random Forest Classifier and experimenting with different preprocessing modifications. I believe that making changes to the dropout layers, trying out various activation functions, and adjusting the number of layers and neurons could also contribute to optimizing the model and achieving the desired goal of 75% accuracy.