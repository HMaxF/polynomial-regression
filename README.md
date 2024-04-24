# Polynomial Regression
Simple Polynomial Regression demo using Python code in Jupyter Notebook  

# Steps:
1. Get the data to make the model
2. Analyze the correlation features of the data, remove uncorrelated features.
   - Eg: in house price prediction, the total door and total windows may not have strong correlation therefore these features can be removed.
3. Split data for prediction and for testing
   - To evaluate the model we should not use data from other source, we only trust all data points from this same source.
   - To create unseen data to get performance of the model using unseen data later.
4. Find optimal Polynomial degree (prediction trial)
   - Polynomial degree parameter is start from 2
   - Do multiple trials to build the model and predict
   - Check and compare the R2 score to get the highest R2 score
5. Predict training data.
6. Predict unseen test data.
7. Compare performance
   - If the prediction result of the training data and unseen test data are close (similar) then the model is good.
   - If the prediction result is far (a lot of gaps) then the model maybe overfit, need to reduce 'degree' value.
8. Extra: Find outlier (data point that is noticeably different from the rest which may be considered as "invalid" data)
   - Using Mean Absolute Error (MAE) & Threshold to find outliers.
   - If we found outliers then we can do further analysis if we need to remove them to improve model.

# Requirement
- Jupyter notebook (https://jupyter.org/)
- Python 3.8+
- Matplotlib

# How to run
- In command line, example: python3 app_polynomial_regression_v1.py random_polynomial_house_prices.csv "[[50,1],[50,5],[50,10]]"
- In Jupyter notebook: app_polynomial_regression_v1.ipynb

# Notes
- The .ipynb has full python code similar to the .py
- Load .ipynb either in VS Code or in browser using Jupyter Notebook
- Inside .ipynb there is Matplotlib plots in 3D (3-axis values) to visualize the data for better understanding of Polynomial Regression.

# Reference:
https://quick.work/?page=view-blog&id=50&title=Polynomial+Regression+in+Python
