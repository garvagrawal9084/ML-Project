from setuptools import find_packages , setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path : str) -> List[str]:
    '''
    this function fill return the list of requirements
    '''

    requirement = []

    with open("requirements.txt") as file :
        requirement = file.readlines()
        requirement = [req.replace("\n" , "") for req in requirement]

        if HYPEN_E_DOT in requirement:
            requirement.remove(HYPEN_E_DOT)

    return requirement

setup(
    name="mlProject" ,
    version="0.0.1",
    author="Garv Agrawal",
    author_email="garvagrawal9084@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)