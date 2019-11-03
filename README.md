# Run Instructions
### Preparations
As briefly mentioned in our report and presentation, we had some difficulties getting the Orange3 library to work properly. There is several ways of installing it. We recommend using Linux in combination with Anaconda. 

If Anaconda is not yet installed on your system, please follow these instructions:
Instructions on how to install Anaconda can be found [here](https://docs.anaconda.com/anaconda/install/linux/). 

## Clone the code
Please clone this repository and navigate to it:
```
git clone https://github.com/JoppeBoekestijn/Arguing-Agents.git
cd Arguing-Agents/
```
## Create conda environment
Please run the following commands to create the necessary `conda` environment:
```
conda config --add channels conda-forge
conda create -n arg python=3.7.4
conda activate arg
```
## Install packages
Then install the required Python packages using `pip`:
```
pip install -r requirements.txt
```
## Install Orange & Jupyter
Now, install `Orange3` and `Jupyter Lab` with all their respective dependencies:
```
conda install orange3
conda install -c defaults pyqt=5 qt
conda install -c conda-forge jupytext
conda install -c anaconda jupyter
conda install -c conda-forge jupyterlab
```
## Start it
Lastly, we need to convert `main.py` to `main.ipynb` with `jupytext`:
```
jupytext --to ipynb main.py
```
Running the code is now easy by launching `Jupyter Lab` and opening the `main.ipynb` file:
```
jupyter lab
```
Run all the cells 
