# House Prediction Deep Learning Model
## Introduction
The Objective of this project is to use linear regression to find the median value of owner-occupied homes in 1000 USD's. A machine learning model is built using. tensorflow.keras API.

## Data
The dataset for this project comes from the real estate industry in Boston (US). This database contains 14 attributes. The target variable refers to the median value of owner-occupied homes in 1000 USD's. The columns in the dataset are defined below:
- CRIM: per capita crime rate by town
- ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- NOX: nitric oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per 10,000 USD
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: lower status of the population (%)
- MEDV: Median value of owner-occupied homes in 1000 USD's (Target)

## Data Preprocessing
Relevant libraries such as pandas, numpy, matplotlib were all loaded for data preprocessing. Before building any machine learning model, we always separate the input variables and output variables. Like here in this data, we are trying to predict the price of a houce, so this is our target column i.e. 'MEDV' and by convention input variables are represented with 'X' and output variables are represented with 'y'.

We split 80% of the data to the training set while 20% of the data to test set using below code. The test_size variable is where we actually specify the proportion of the test set.

By passing our X and y variables into the train_test_split method, we are able to capture the splits in data by assigning 4 variables to the result.

## Training the Model
After splitting the data into training and testing sets, it's time to train our first deep learning model.
The 5 Step Model Life-Cycle
A model has a life-cycle, and this very simple knowledge provides the backbone for both modeling a dataset and understanding the tf.keras API.

#### The five steps in the life-cycle are as follows:

- <strong> Define the model.</strong><br>
Defining the model requires that you first select the type of model that you need and then choose the architecture or network topology.
From an API perspective, this involves defining the layers of the model, configuring each layer with a number of nodes and activation function, and connecting the layers together into a cohesive model.<br>
Models can be defined either with the Sequential API or the Functional API (you will know this in later modules). Here we will define the model with Sequential API.

- <strong> Compile the model.</strong><br>
Compiling the model requires that you first select a loss function that you want to optimize, such as mean squared error or cross-entropy.
It also requires that you select an algorithm to perform the optimization procedure. We’re using RMSprop as our optimizer here. RMSprop stands for Root Mean Square Propagation. It’s one of the most popular gradient descent optimization algorithms for deep learning networks. RMSprop is an optimizer that’s reliable and fast.

- <strong> Fit the model.</strong><br>
Fitting the model requires that you first select the training configuration, such as the number of epochs (loops through the training dataset) and the batch size (number of samples in an epoch used to estimate model error).
Training applies the chosen optimization algorithm to minimize the chosen loss function and updates the model using the backpropagation (don't worry if you don't know this term, you will know it in the next module) of error algorithm.<br>
Fitting the model is the slow part of the whole process and can take seconds to hours to days, depending on the complexity of the model, the hardware you’re using, and the size of the training dataset.<br>
From an API perspective, this involves calling a function to perform the training process. This function will block (not return) until the training process has finished. While fitting the model, a progress bar will summarize the status of each epoch and the overall training process.

- <strong> Evaluate the model.</strong><br>
Evaluating the model requires that you first choose a holdout dataset used to evaluate the model. This should be data not used in the training process i.e. the X_test.
The speed of model evaluation is proportional to the amount of data you want to use for the evaluation, although it is much faster than training as the model is not changed. From an API perspective, this involves calling a function with the holdout dataset and getting a loss and perhaps other metrics that can be reported.<br>

## Summary of hyperparameter tuning
Most machine learning problems require a lot of hyperparameter tuning. Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly. You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:<br>

- Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
- If the training loss does not converge, train for more epochs.
- If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
- If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
- Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
- Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
- For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.<br>
Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.

We can do a hyperparameter tuning procedure in two ways:

<strong>Implementing hyperparameter tuning with Sklearn</strong><br>
<strong>Implementing hyperparameter tuning with Keras</strong><br>

### Implementing hyperparameter tuning with Sklearn<br>
Well, we can automate the hyperparameter tunning using GridSearCV. GridSearchCV is a hyperparameter search procedure that is done over a defined grid of hyperparameters. Each one of the hyperparameter combinations is used for training a new model, while a cross-validation process is executed to measure the performance of the provisional models. Once the process is done, the hyperparameters and the model with the best performance are chosen.

Let's first take a look at the implementation of GridSearchCV with Sklearn, following the steps:

- Define the general architecture of the model
- Define the hyperparameters grid to be validated
- Run the GridSearchCV process
- Print the results of the best model

### Implementing hyperparameter tuning with Keras<br>
Now we will go through the process of automating hyperparameter tuning using Random Search and Keras. Random Search is a hyperparameter search procedure that is performed on a defined grid of hyperparameters. However, not all hyperparameter combinations are used to train a new model, only some selected randomly, while a process of cross-validation to measure the performance of temporal models. Once the process is complete, the hyperparameters and the best performing model are chosen.

Let's take a look at the implementation of Random Search with Keras, following the steps:

- Install and import all the packages needed
- Define the general architecture of the model through a creation function
- Define the hyperparameters grid to be validated
- Run the GridSearchCV process
- Print the results of the best model

To execute the hyperparameter tuning procedure we will use the keras-tuner, a library that helps you pick the optimal set of hyperparameters for your TensorFlow model.

## Make a Prediction<br>
Making a prediction is the final step in the life-cycle. It is why we wanted the model in the first place.

It requires you have new data for which a prediction is required, e.g. where you do not have the target values.

From an API perspective, you simply call a function to make a prediction of a class label, probability, or numerical value: whatever you designed your model to predict.
