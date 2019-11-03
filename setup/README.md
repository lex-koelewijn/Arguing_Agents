# Instructions
Using the Orange package for CN2 has proven to be quite difficult, as the documentation is lacking, and the package is getting deprecated.
```
We have created a requirements.txt file with the correct packages to install via pip, but we found out that Orange3 doesn't work if we install it via pip. Therefore we also provide a way to run our program using a conda environment, which is tested and working on Linux and MacOS.
```
For pip install:
```
pip install -r setup/requirements.txt
```
For anaconda install:
```
For installing anaconda on Linux: https://problemsolvingwithpython.com/01-Orientation/01.05-Installing-Anaconda-on-Linux/
For installing anaconda on MacOS: https://docs.anaconda.com/anaconda/install/mac-os/
```
In the terminal create an environment:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
```
conda create --name myenv python=3.7.4              [where myenv is a name to be specified yourself]
```
conda activate myenv                                [where myenv is name specified for conda environment]
```
Now install the correct packages by running in the terminal
```
conda install pandas
```
conda install numpy
```
conda install orange3
```
conda install matplotlib
```
conda install jupyter
```
conda install scikit-learn
```
Navigate to /Arguing-Agents/ and type:
```
jupyter notebook
```
Now open the `main.ipynb` file and run the cells in order to run our code
```
We have also supplied a python file with the same code, but this is converted from the notebook file, so it might not work if it is run from the terminal
