# Saved Models

This directory contains all the saved models for the Chess Engine project.

## Overview

The `savedModels` directory is used to store the trained models for the Chess Engine. All the models will be saved in this directory for easy access and reuse. The lower the score in `bestModel.txt` the better the Chess Engine is.

## File Structure

The `savedModels` directory follows a specific file structure to organize the models. Each model is saved in its own file, named after the model's unique identifier. 

## Adding Models

To add a new model, follow these steps:

1. Create a new model inside the `savedModels` directory.
2. Unique identifier for the model is automaticsly made by the program.
3. All the necessary files and resources related to the model are made with `main.py`.
4. After training some new games with `main.py`, only the better model of the lastest models made in the process will be saved in the `savedModels` directory.


## Accessing Models

To access a specific model, navigate to the corresponding subdirectory inside the `savedModels` directory. Inside each `bestModel.txt`, you will find the files and resources associated with that model. And if you want to try a model vs Stockfish elo use `play.py`.

