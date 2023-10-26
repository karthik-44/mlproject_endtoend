# End to End Machine Learning Project  

The main objective of this project is to implement a general machine learning work flow using python scripts.  
These can be combined as a package and be deployed to any cloud environment or to pypi.  


## creation and Activation of virtual environment
```bash
(base) E:\mlproject_endtoend>conda create -p venv_mlee python==3.9 -y
(base) E:\mlproject_endtoend>conda activate venv_mlee/
```

Mainly we shall use git for version controlling of the code.  

## Version control  
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


