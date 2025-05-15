from setuptools import setup, find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name='autonote',
    description="Code snippet manager for Jupyter Notebook & Notebook enhancements + data science helper functions",
    packages=find_packages(),  # It will find all packages in your directory
    install_requires=requirements  # This is the key line to install dependencies
)
