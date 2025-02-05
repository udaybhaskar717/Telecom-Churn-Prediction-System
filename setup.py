from setuptools import find_packages, setup
from typing import List
def get_requirements() -> List[str]:
    """
    This function will read the requirements.txt file and return a list of requirements.
    """
    requirements: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()  # Remove leading/trailing whitespace
                if line and not line.startswith('#') and line != '-e .':
                    requirements.append(line)
    except FileNotFoundError:
        print("requirements.txt file not found")
    return requirements

setup(
    name="customer_churn",
    version="0.0.1",
    author="Uday",
    author_email="udaybhaskar717@gmail.com",
    install_requires=get_requirements(),  # Use install_requires instead of requires
    packages=find_packages()
)