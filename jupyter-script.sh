#!/bin/bash

# This converts the .py to .ipynb
# Start the jupyter server
# Converts .ipynb back to .py once the server is closed

jupytext --to ipynb main.py
jupyter lab
jupytext --to py main.ipynb
