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
# from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import pickle # to save/load model to/from file

import argparse

def buildModelBasedOnCSV(csv_filepath):

    try:
        df = pd.read_csv(csv_filepath)

        # define X ==> data to train (independent) and y ==> the target (dependent, annotated)
        X = df.drop(df.columns[-1], axis=1) # numpy.ndarray
        y = df.take([-1], axis=1)

        return df
    except FileNotFoundError:
        print(f"Error, {csv_filepath} is not found")

    return None


def createModels():

    print('*** createModels()')

    # create model by fitting (train) the prepared data
    global modelLR
    modelLR = LinearRegression()


    """
    problem: got warning message
    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/utils/validation.py:1229: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    y = column_or_1d(y, warn=True)

    solution: because y_GroundTruth is 1d then add .values.ravel()
    """
    # remove title from X, only use the .values (numerical) because y is also values
    modelLR.fit(x_Features.values, y_GroundTruth.values.ravel())

    # using column name, it will also only get the 'values'
    #modelLR.fit([x_Features['location'], x_Features['size'], x_Features['age']].values, y_GroundTruth)

    # show the coefficient and intercept value for the Linear Regression !
    print(f"coef: {modelLR.coef_}")
    print(f"intercept: {modelLR.intercept_}")

    global modelSVM
    modelSVM = SVC()
    # remove title from X, only use the .values (numerical) because y is also values
    modelSVM.fit(x_Features.values, y_GroundTruth.values.ravel())

    global modelDTC
    modelDTC = DecisionTreeClassifier()
    # remove title from X, only use the .values (numerical) because y is also values
    modelDTC.fit(x_Features.values, y_GroundTruth.values.ravel())

    save_model()

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

    min_r2_score = 0.99 # if more than maybe overfit

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
    #print(f"\nr2 score = {a_r2_score}, degree = {degree}\n***")
    print(f"\nHighest r2 score = {highest_r2_score}, degree = {degree_highest_r2_score}\n***")
    print(f"\ntarget and predicted")
    print(y_train, y_poly_pred.astype(int))

    return degree_highest_r2_score, poly

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
    print(f"   1st argument is a filename, either '.csv' or '.model'.")
    print(f"      If the filename is '.csv' then this app will create a model and save it to a file with the same filename '.model'")
    print(f"      If the filename is '.model' then this app will load the model (without building).")
    print(f"   2nd argument is a data to predict, eg: \"4,5,6,7\" and will generate the prediction value.")
    print(f"\n\nExample: mydata.csv \"[[1,25], [2,38], [2,29]]\"")

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