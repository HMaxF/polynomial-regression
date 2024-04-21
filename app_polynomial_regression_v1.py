"""
Application to build Machine Learning model based on input data.

App v0.98

Hariyanto Lim
"""
import sys # to get arguments (parameter) manually

import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error # == [(a - b) ** 2]
from sklearn.metrics import mean_absolute_error # == [abs(a - b)]

import pickle # to save/load model to/from file

import argparse

def save_model(model, poly, filename):
    print('*** save_model()')

    # save the model and the poly to disk
    pickle.dump((model, poly), open(filename, 'wb'))

def load_model(filename):    

    try:
        # load both model and poly
        model, poly = pickle.load(open(filename, 'rb'))

        print(f"*** load_model(): '{filename}' loaded")

        return model, poly

    except FileNotFoundError:

        print(f"*** load_model(): '{filename}' not found")
        return None

    return None


def create_mapped_text_to_csv(original_filename, df):
    """
    prepare and cleaning data before training
    NOTES:
    1. in Machine Learning, many times are spent to prepare and clean data.
    2. Below is only an example of "cleaning" and "preparation"
    """

    """
    Show how to display all uniques value of a column
    """
    print(f"Example to convert 'text' to integer value")

    unique_text = df['Area'].unique()        

    # create DataFrame to map text as integer
    df_dict = pd.DataFrame()
    df_dict['text'] = unique_text
    df_dict['integer'] = pd.Series(unique_text).astype('category').cat.codes.values
    print(df_dict)

    """
    Convert unique text to integer value
    """
    # prepare and cleaning data before training
    # NOTE: in Machine Learning, many times are spent to prepare and clean data.

    # add a new column 'brand_int' to represent 'brand' string, the value is start from 1 (not 0)
    df['area_int'] = df['Area'].rank(method='dense', ascending=True).astype(int)

    """
    Optional to save mapped value to separated CSV
    """
    # get mapped values
    mapped_text = df[['Area','area_int']]

    # keep unique only
    mapped_text.drop_duplicates(inplace=True) # may generate warning, can be ignored

    # save mapped brand to user, so user can understand it
    mapped_text.to_csv(original_filename + "_dictionary_mapped_text_to_integer.csv", index=False)

    """
    Fix Dataframe
    """    
    df.drop(['Area'], axis=1, inplace=True)

    # Get the last column name
    last_column_name = df.columns[-1]

    # Extract the last column
    last_column = df.pop(last_column_name)

    # Insert the last column at the beginning
    df.insert(0, last_column_name, last_column)

    # rearrange column
    #df = df[['area_int', 'model_year', 'milage', 'price']]

    # save to a new CSV
    df.to_csv(original_filename + '_fixed.csv', index=False)

# function to build optimal model
def create_polynomial_regression_model(df):

    # define X ==> data to train (independent) and y ==> the target (dependent, annotated)
    X = df.drop(df.columns[-1], axis=1) # df.columns[-1] == the most right side column
    y = df.take([-1], axis=1)

    # check the content of X, all columns should be numerical (int or float), can not be string !!
    numeric_df = df.apply(pd.to_numeric, errors='coerce')
    is_all_numeric = not numeric_df.isnull().values.any()
    if is_all_numeric == False:
        print(f"the data content of CSV to be used as training model has non-numeric value, please fix this!")
        print(df)

        return None

    modelLR = LinearRegression()

    # print(X)
    # print(y)

    degree, poly = find_optimal_polynomial_degree(modelLR, X, y)

    print(f"Polynomial Regression model is created with degree = {degree}")

    return modelLR, poly

def find_optimal_polynomial_degree(modelLR, X_train, y_train):

    r2_score_lower_limit_threshold = 0.1
    degree_highest_r2_score = 0
    highest_r2_score = 0

    degree = 2 # start value NOTE: first loop the value should be >= 2

    a_r2_score = 0

    min_r2_score = 0.90 # if more than maybe overfit

    y_poly_pred = None

    # EXAMPLE logic to find optional 'degree' ONLY up to 10
    while degree <= 10 and a_r2_score < min_r2_score: 
        """
        WARNING: the higher the 'degree' value, the SLOWER the prediction !!!
        """
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        # transform X_train value to poly        
        X_train_poly = poly.fit_transform(X_train.values)

        # train the transformed (using PolynomialFeatures) data
        # with only values (exclude header text)
        modelLR.fit(X_train_poly, y_train.values)

        # predict the training data to see r2_score
        y_poly_pred = modelLR.predict(X_train_poly)
        
        # NOTE: in this demo, we only use r2_score() to check the accuracy
        a_r2_score = r2_score(y_train, y_poly_pred)
        if a_r2_score < min_r2_score:
            print(f"degree = {degree} only got r2 score = {a_r2_score}")
        
        if a_r2_score > highest_r2_score:
            # save it
            highest_r2_score = a_r2_score
            degree_highest_r2_score = degree
        elif a_r2_score < highest_r2_score - r2_score_lower_limit_threshold:
            """
            IF r2_score is smaller when using higher 'degree'
            THEN it means accuracy went down
            NORMALLY it won't go up again, so break loop
            """ 
            print(f"degree={degree}'s r2 score is lower than previously, so break loop")
            break

        degree += 1 # increase trial counter

    

    #################################################
    # recreate the model and the poly using the degree_highest_r2_score 
    # because current model and poly may be different than HIGHEST R2 model & poly.

    poly = PolynomialFeatures(degree=degree_highest_r2_score, include_bias=False)

    # transform X_train value to poly        
    X_train_poly = poly.fit_transform(X_train.values)

    # train the transformed (using PolynomialFeatures) data
    # with only values (exclude header text)
    modelLR.fit(X_train_poly, y_train.values)
    #################################################

    # show the coefficient and intercept value
    print(f"***\n{modelLR.coef_ = }\n{modelLR.intercept_ = }")
    print(f"\nHighest r2 score = {highest_r2_score}, degree = {degree_highest_r2_score}\n***")

    ### get mean value of error 

    # change to 1D and convert to int
    y_poly_pred_flat = y_poly_pred.flatten().astype(int) 
    print(f"Error rate: {mean_absolute_error(y_train['Price'], y_poly_pred_flat)}")

    ### debugging: compare and find outlier
    temp = find_outlier_as_table(y_train['Price'], y_poly_pred_flat, 50)
    pd.set_option('display.max_rows', None)  # Set max_rows to None to display all rows  
    print(temp) #display

    return degree_highest_r2_score, poly

def find_outlier_as_table(target, predicted, threshold):
    """
    outlier is data point that stands out from the rest of the group.
    It's either much higher or lower than most of the other data points in the set.

    Create a DataFrame containing target, predicted, MAE, and outlier flag.
    Parameters:
    target (list or array-like): The array of actual values.
    predicted (list or array-like): The array of predicted values.
    threshold (float): The threshold value for detecting outliers.

    Returns:
    pandas.DataFrame: DataFrame containing target, predicted, MAE, and outlier flag.
    """
    # Calculate Mean Absolute Error (MAE)
    mae = [abs(t - p) for t, p in zip(target, predicted)]
    
    # Determine outliers
    outliers = [1 if error > threshold else 0 for error in mae]
    
    # Create DataFrame
    df = pd.DataFrame({
        'target': target,
        'predicted': predicted,
        'mae': mae,
        'outlier': outliers
    })
    
    return df

def load_csv_into_dataframe(csv_filename):
    try:
        df = pd.read_csv(csv_filename)

        return df

    except FileNotFoundError:
        print(f"*** load_csv_into_dataframe(): '{csv_filename}' not found")

    return None

def show_usage():
    print(f"Simple Polynomial Regression")
    print(f"positional arguments:")
    print(f"   1st argument is a filename with these file extension:")
    print(f"      If the extension is '.csv' then this app will create a model and save it to a file with the same filename with extension '.model'")
    print(f"      If the extension is '.model' then this app will load the model (without building).")
    print(f"   2nd argument is a data-to-predict that either a single value array or multiple value array.")
    print(f"\n\nExample:")
    print(f"   mydata.csv \"[[76]]\"")
    print(f"   mydata.model \"[[10],[20],[30],[40]]\"")
    print(f"   mydata.csv \"[[1,25], [2,38], [2,29]]\"")

if __name__ == "__main__":

    """
    # get parameter using 'argparse'
    parser = argparse.ArgumentParser(description='Polynomial  Regression Model Trainer')
    parser.add_argument('filename', type=str, help='Path to the CSV file')
    parser.add_argument('predict', type=str, help='Data to predict')
    args = parser.parse_args()

    # check filename
    if args.filename == None:
        # print(f"parameter 'filename' not found")
        show_usage()
        exit(-1)
    """

    # manually parse the arguments
    if len(sys.argv) != 3:
        show_usage()
        exit(-1)
    
    # Extract the filename from the command-line arguments
    #input_filename = args.filename    
    input_filename = sys.argv[1]
    data_str = sys.argv[2]

    operation = None
    if input_filename.endswith(".csv"):
        operation = 'csv'
    elif input_filename.endswith(".model"):
        operation = 'model'
    else:
        show_usage()
        exit(-2)
    
    if operation == 'csv':
        df = load_csv_into_dataframe(input_filename)
        if df is None:
            print(f"load csv '{input_filename}' failed")
            exit(-3)

        #print(df)
        
        model, poly = create_polynomial_regression_model(df)
        if model is None:
            # NOTE: TESTING only !!
            create_mapped_text_to_csv(input_filename, df)
        else:
            # get input filename, replace extension '.csv' to '.model'
            model_filename = input_filename[0:-4] + '.model'
            save_model(model, poly, model_filename)

    else:
        # option is 'model'
        model, poly = load_model(input_filename)
        if model is None:
            print(f"error, failed to load model")
            exit(-4)

    # convert string to numpy.array
    js = json.loads(data_str)
    data_to_predict = np.array(js)
    print(f"data to predict, shape: {data_to_predict.shape}: {data_to_predict}")

    # reshape data_to_predict
    #reshaped_data = data_to_predict.reshape(1,-1)

    # apply the same transformation to the data to be predicted, to ensure the data is in the same form as learned data ('model')
    data_to_predict_poly = poly.transform(data_to_predict)

    # predict
    #result = model.predict(reshaped_data)
    #result = model.predict([[1,50,2020,12]])
    result = model.predict(data_to_predict_poly)

    print(f"prediction result:")

    # attempt to display number without 'e+' notation 
    #np.set_printoptions(precision=3, suppress=True) 
    #result = np.round(result, decimals=4)
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    #result = [f'{num:.0f}' for num in result]
    #print(f"{result:.0f}")
    #print(result)
    #result = f"{result:.2f}"
    print(result)

    print(f"*** app is ended normally ***")