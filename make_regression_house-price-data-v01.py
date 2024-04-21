import numpy as np
import pandas as pd

def get_random():
    # generate random value between 1.0 to 1.5
    return 1 + (np.random.rand() * 0.5)

# Define a mapping dictionary
city_mapping = {'Seoul': 1, 'Incheon': 2, 'Busan': 3}

# Define cities and their corresponding price factors
city_prices = {'Seoul': 10, 'Incheon': 7, 'Busan': 4}

# Define the number of rows for each city
n_rows_per_city = 30

# Generate random values for size and age for each city
np.random.seed(42)  # For reproducibility
data = []
for city, price_factor in city_prices.items():
    for _ in range(n_rows_per_city):
        size = np.random.randint(30, 101)  # Size in square meters
        age = np.random.randint(1, 11)  # Age in years

        # Price formula
        price = (size * price_factor) # base
        price += (get_random() * price) # random addition
        #price -= ((get_random() * age) * (get_random() * 5)) # subtract age
        price -= (get_random() * age * 3) # subtract age

        price = int(price) # remove decimal

        data.append([city_mapping[city], size, age, price])

# Create a DataFrame from the generated data
df = pd.DataFrame(data, columns=['City', 'Size', 'Age', 'Price'])

# # Sort the DataFrame by city then size
df = df.sort_values(by=['City', 'Size']).reset_index(drop=True)



# Print the final dataframe
print(df)


df.to_csv('random_house_prices.csv', index=False)