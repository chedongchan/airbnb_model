Documentation for Airbnb Property Listing Model


> Milestone 3: Data Preparation

In this milestone, I have observed and processed the tabular data into a format that can be used to train/test models.
All empty cells were dropped. Made sure that the descriptions are of an appropriate string format. Created a tuple that contains the features and labels. Images were also processed in the prepare_image_class.py file.

Running the 'main_airbnb_data_processing.py' file, one can import the tabular_data_class.py and prepare_image_class.py to prepare the dataset into a format appropriate for machine learning for the next steps.

The final csv file is clean_tabular_data.csv,  and load_airbnb_data function will return a tuple in the features,labels format. 


> Milestone 4: Create a regression model

In this milestone, I have set up a modelling_class.py script that contains the steps to create a regression model. Namely, using the SGD regressor module. In this module I have learnt what machine learning model types there are and why some are more appropriate for analysis than other. 
I have designed my own hyperparameter tuning function but also used the scikit-learn module (GridSearchCV). I used the K-Fold Validation to see if the differences seen between the first and second best models are statistically signficant. I have beaten the base linear regression by tuning for hyperparameters.

> Milestone 5: Create a classification model

Much like the previous milestones, I implemented different classification models such as random forests, gradient boosting and decision trees to see if I can beat the baseline classification model. 

> Milestone 6: Create a configurable neural network

Here I learnt about neural networks. I used a new tool called PyTorch which allows me to create objects that hold tensor gradient values. In order to train my model, I had to process the data into a custom PyTorch dataset, which is then loaded into the model using DataLoader. Then finally, these will go through a training loop that trains the model based on training neural network functions. For my first model, I use a basic Linear regression model with 2 layers and a ReLU activation function. I tuned the model using a variety of parameters configuring learning rates and number of hidden layers.


Finally, to describe my neural network model - Here is the abstract.


>The PriceNightDataset class is a custom PyTorch dataset class that is used to load and preprocess the input data for training or evaluating the neural network. It takes two arguments in the constructor: X and y. These are the input features and output labels for the training or test data.

The __len__ method returns the length of the dataset, which is the number of examples in the dataset.

The __getitem__ method is called when an example from the dataset is accessed. It takes an index as an argument and returns the input and output data for the example at that index.

The generate_nn_configs function generates a list of config dictionaries that contain hyperparameters for the neural network. It takes an integer argument num_configs which specifies the number of config dictionaries to generate. The config dictionaries contain the learning rate and hidden size hyperparameters.

The train function trains the neural network model. It takes four arguments: X_train and y_train which are the input features and output labels for the training data, a config dictionary containing the hyperparameters for the model, and a train_loader which is a PyTorch DataLoader for the training data. The function defines the model architecture, optimizer, and loss function, and then trains the model for 100 epochs using the training data.

The evaluate function evaluates the model on the test set. It takes four arguments: the model to evaluate, X_test and y_test which are the input features and output labels for the test data, and a test_loader which is a PyTorch DataLoader for the test data. The function defines the loss function and then calculates the loss and R^2 value for the test set using the model's predictions and the true labels.

The run_experiment function trains and evaluates a neural network model with a given config. It takes three arguments: X_train, y_train, and X_test which are the input features for the training and test sets, and y_test which is the output label for the test set. The function generates a train_loader and test_loader using the PriceNightDataset class, trains the model using the train function, and evaluates the model using the evaluate function. It returns the model, loss, and R^2 value for the test set.

The run_experiments function runs multiple experiments with different configs and stores the results in a list. It takes five arguments: X_train, y_train, X_test, y_test, and num_experiments. The function generates a list of config dictionaries using the generate_nn_configs function, and then runs an experiment for each config using the run_experiment function. It stores the results of each experiment in a list of dictionaries, which includes the config, loss, and R^2 value for each experiment. Finally, it returns the list of results


