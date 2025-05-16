from setuptools import setup, find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(
    name='text_lense',
    version='0.1.0',
    author='Jan du Plessis',
    description="Free-text analysis tool for unstructured text based surveys.",
    packages=find_packages(),  # It will find all packages in your directory
    install_requires=requirements  # This is the key line to install dependencies
)
