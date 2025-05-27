from setuptools import setup, find_packages

setup(
    name='dataprep',                  
    version='0.1.0',                
    description='Custom Data Transformers for Pandas & Sklearn Pipelines',
    author='kaiku',
    author_email='mandarioalexis@email.com',
    packages=find_packages(),          # Automatically find your modules
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    python_requires='>=3.7',          
)
