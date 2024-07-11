# ProsperIndex
# Global Country Development & Prosperity Index Prediction using Machine Learning

## Description
This project aims to predict the Prosperity Index of countries using various regression techniques. The dataset used is the "2023 Global Country Development & Prosperity Index".
The goal is to build a robust predictive model with high accuracy.

## Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

## Files in the Repository
- `README.md`: Overview of the project and instructions.
- `data/`: Contains the dataset used for the project.
- `code.py`: Python script with data exploration, preprocessing, modeling, and evaluation steps.
- `models/`: Serialized models and results.
- `visualizations/`: Plots and graphs generated during data exploration and model evaluation.

## Summary of Results
The best-performing model was found to be Gradient Boosting Regression, which achieved the lowest RMSE and highest R² on the validation set. Key features impacting the Prosperity Index included GDP per capita, education levels, and healthcare access.

## Installation Instructions
1. Clone the repository: `git clone https://github.com/username/global-prosperity-index.git`
2. Navigate to the project directory: `cd global-prosperity-index`
3. Install the required libraries: `pip install -r requirements.txt`

## Usage
1. Run the script to see the data exploration, preprocessing, and modeling steps.
2. Load the trained models from the `models/` directory to make predictions on new data.

## Related post
https://medium.com/@daniel.garciad/predicting-global-country-development-prosperity-index-using-machine-learning-44bdd6d29fb3

## About the Author & Acknowledgments
This project was developed by Daniel Garcia as part of his course for Udacity.
We would like to express our gratitude to the person who added this dataset  in KAGGLE so I can used for this work. 
Additionally, we thank the open-source community for developing and maintaining the libraries and tools that made this project possible.
