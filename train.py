from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.run import Run

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

import argparse
import os
import numpy as np
import joblib
import pandas as pd

# Imports dataset
mbdataset = TabularDatasetFactory.from_delimited_files("https://raw.githubusercontent.com/czofficial/nd00333-capstone/4a6a4924bdd4c6a188aeb24e0c282bae11c8933b/mercedes.csv")

df = mbdataset.to_pandas_dataframe()

# Cleans dataset
def clean_data(df):
   
    x_df = df
    x_df = pd.get_dummies(df, columns=['model', 'transmission', 'fuelType'])
    y_df = x_df.pop("price")

    return x_df,y_df

x, y = clean_data(df)

# Splits dataset into train and test
x_train,x_test,y_train,y_test = train_test_split(x,y)

run = Run.get_context()

def main():
    # Adds arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_depth',
                        type=int,
                        default=1,
                        help="The maximum depth of the tree.")
    parser.add_argument('--min_samples_split',
                        type=int,
                        default=2,
                        help="The minimum number of samples required to split an internal node.")
    parser.add_argument('--min_samples_leaf',
                        type=int,
                        default=1,
                        help="The minimum number of samples required to be at a leaf node.")

    args = parser.parse_args()

    run.log("max_depth:", np.int(args.max_depth))   
    run.log("min_samples_split:", np.int(args.min_samples_split))
    run.log("min_samples_leaf:", np.int(args.min_samples_leaf))


# Trains random forest
    model = RandomForestRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)

# Calculates MAE
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    run.log('mae', np.int(mae))

# Saves the model   
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/hd-model.pkl')
    
if __name__ == '__main__':
    main()