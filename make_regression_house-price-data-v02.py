import numpy as np
import pandas as pd

def get_random():
    # generate random value between 0.5 to 1.5
    return 0.5 + (np.random.rand() * 1.0)

# Set random seed for reproducibility
np.random.seed(29)

# Define the number of data points
num_data_points = 100

# Generate random sizes and ages
sizes = np.random.randint(30, 101, num_data_points)
ages = np.random.randint(1, 11, num_data_points)

# Simulate price with a polynomial relationship, use higher exponent values 
prices = 10 * (sizes ** 2) # base 
prices -= (ages ** 3) # reduce by age (the older the cheaper)

prices = prices * get_random() # numpy array multiple by single value

#prices /= 1000 # reduce the price (too big is too difficult to see and quick-calc)
prices = prices.astype(int) # remove decimal

# Create a pandas DataFrame
df = pd.DataFrame({'Size (m²)': sizes, 'Age (years)': ages, 'Price': prices})

# Sort the DataFrame by size then by age
df = df.sort_values(by=['Size (m²)', 'Age (years)']).reset_index(drop=True)

# Display the DataFrame
print(df.to_string())

df.to_csv('random_polynomial_house_prices.csv', index=False)