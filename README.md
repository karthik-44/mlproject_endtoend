# End to End Machine Learning Project  

The main objective of this project is to implement a general machine learning work flow using python scripts.  
These can be combined as a package and be deployed to any cloud environment or to pypi.  


## creation and Activation of virtual environment
```bash
(base) E:\mlproject_endtoend>conda create -p venv_mlee python==3.9 -y
(base) E:\mlproject_endtoend>conda activate venv_mlee/
```

Mainly we shall use git for version controlling of the code.  

## Version Control  
```bash
(E:\mlproject_endtoend\venv_mlee) E:\mlproject_endtoend>  
git config --global user.email “<github_email”
git config --global user.name “<github_username>”

git init . 
echo “ml end to end” >> README.md
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/karthik-44/mlproject_endtoend.git
git push -u origin main
```

Create a .gitignore file. This doesn't track files listed in it for version tracking.  

We shall periodically commit and push so that the code changes are tracked and updated to the Github repo.  

Add the following files:  
- setup.py - Creates a machine learning application as a package.
- requirements.txt - this contains the list of packages that we might need for the machine learning project.

```bash
pip install -r requirements.txt
```

This installs the packages in the environment and also creates a folder called mlproject_endtoend.egg-info. (can be used to deploy the package in pypi if intended.)  

Now we shall create a folder **src** where we work most of the time for our machine learning project. The **src** folder structure is generally of the following structure: 
```bash
src
 ┣ components
 ┃ ┣ __pycache__
 ┃ ┃ ┣ data_transformation.cpython-39.pyc
 ┃ ┃ ┣ model_trainer.cpython-39.pyc
 ┃ ┃ ┗ __init__.cpython-39.pyc
 ┃ ┣ data_ingestion.py
 ┃ ┣ data_transformation.py
 ┃ ┣ model_trainer.py
 ┃ ┗ __init__.py
 ┣ pipeline
 ┃ ┣ __pycache__
 ┃ ┃ ┣ predict_pipeline.cpython-39.pyc
 ┃ ┃ ┣ predict_pipeline_krish.cpython-39.pyc
 ┃ ┃ ┗ __init__.cpython-39.pyc
 ┃ ┣ predict_pipeline.py
 ┃ ┣ train_pipeline.py
 ┃ ┗ __init__.py
 ┣ __pycache__
 ┃ ┣ exception.cpython-39.pyc
 ┃ ┣ logger.cpython-39.pyc
 ┃ ┣ utils.cpython-39.pyc
 ┃ ┗ __init__.cpython-39.pyc
 ┣ exception.py
 ┣ logger.py
 ┣ utils.py
 ┗ __init__.py

```

The main elements from **src** folder are :  
### exception.py
When we code, we generally gets errors. We can customize the error messages so that it will be in a particular format and be informational. This way saves a lot of time over the course of the project. So, we shall create **exception.py** file for custom exception handling. We might be interested in at which file, which line the error occurred and the error message.
Raise the custom exception in try-except block in any script of our project.  


### logger.py 
Another important configuration is **logger.py**. This logs the info during execution into the log file that is configured by us.  
Create this script file and call this script from any other scripts in our project to log the details into the log file.  
A sample information that is logged into one of the files:  
```bash
[ 2023-06-29 22:42:37,895] 187 werkzeug - WARNING -  * Debugger is active!
[ 2023-06-29 22:42:37,898] 187 werkzeug - INFO -  * Debugger PIN: 116-420-621
[ 2023-06-29 22:42:39,740] 187 werkzeug - INFO - 127.0.0.1 - - [29/Jun/2023 22:42:39] "POST /predictdata HTTP/1.1" 200 -
[ 2023-06-29 22:43:02,724] 187 werkzeug - INFO - 127.0.0.1 - - [29/Jun/2023 22:43:02] "POST /predictdata HTTP/1.1" 200 -
[ 2023-06-29 22:43:40,075] 187 werkzeug - INFO - 127.0.0.1 - - [29/Jun/2023 22:43:40] "POST /predictdata HTTP/1.1" 200 -
[ 2023-06-29 22:46:23,742] 187 werkzeug - INFO -  * Detected change in 'E:\\mlproject_endtoend\\app_krish.py', reloading

```

### utils.py  
This is another important file where we can define common utility functions that can be used through out the project.  
We shall define the functions here and call them in different scripts. This is done for better code organization and maintenance.  
Some common utility functions are load_object, save_object(for loading and saving the python objects as pkl files) , evaluate_models(to evaluate different models)

### Components
Create a folder components, this contains all the modules related to :
### Data Ingestion
**data_ingestion.py** 

Create the data ingestion configuration such as where to store the raw data, train, test data.
Load the data and split the data into train, test sets for the mentioned locations.  

### Data Transformation
**data_transformation.py**  
The general steps in this file:  

- Pipeline, imputation, preprocessing (standard scalar, one hot encoder), column transformer etc.,. are imported.
- Create the transformation configuration such as where to store the pre-processor pkl file.
- Create separate pipelines for numerical and categorical features.
- Combine them in the column transformer and return the pre-preprocessor object.


### Model Training
**model_trainer.py**  
This file generally has the following steps implemented in it:  
- Create the model configuration such as where to store the model pkl file.
- Having the train, test array we can test on different models.
- Fit different models to our train, validation data and evaluate them on the test data. Even hyper-parameter tuning is a part of this step.
- Get the model performance for various models and compare them.
- Store the best model in the desired location.

### Pipelines
- **train_pipeline.py**: This might not be needed for us because we have implemented the model training in model_trainer.py.
- **predict_pipeline.py**: This contains the information about the pre-processor.pkl, model.pkl file locations. We shall load them and when we pass an input data to the methods defined in this script, we shall get the predicted output from our best trained model.


