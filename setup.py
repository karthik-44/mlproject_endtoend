from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(filepath:str)->List[str]:
    """
    Function to return the required packages as a list
    """
    requirements=[]
    with open(filepath) as filehandle:
        requirements=filehandle.readlines()
        requirements=[req.replace("\n","") for req in requirements] #as we don't want \n to appear in the package names.
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)



setup(
    name="mlproject_endtoend",
    version="0.0.1",
    author="Karthik",
    author_email="kvrk127@gmail.com",
    packages=find_packages(),
    # install_requires=['pandas','numpy','seaborn'] #packages to install
    install_requires=get_requirements('requirements.txt') #packages to install as a list from the function
    
)
