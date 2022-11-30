Documentation for Airbnb Property Listing Model


> Milestone 3: Data Preparation

In this milestone, I have observed and processed the tabular data into a format that can be used to train/test models.
All empty cells were dropped. Made sure that the descriptions are of an appropriate string format. Created a tuple that contains the features and labels. Images were also processed in the prepare_image_class.py file.

Running the 'main_airbnb_data_processing.py' file, one can import the tabular_data_class.py and prepare_image_class.py to prepare the dataset into a format appropriate for machine learning for the next steps.

The final csv file is clean_tabular_data.csv,  and load_airbnb_data function will return a tuple in the features,labels format. 


> Milestone 4: Create a regression model

In this milestone, I have set up a modelling_class.py script that contains the steps to create a regression model. Namely, using the SGD regressor module. I am getting values of RMSE and MSE that are very large and I am having trouble understanding if it is a problem of my dataset or code. I am able to get reasonable results using downloadable datasets so I am leaning towards the possibility that the problem is with my dataset preparation. However, the airbnb dataset is significantly larger with 9/10 numerical features... 



