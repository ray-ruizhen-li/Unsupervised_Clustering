import pandas as pd

# create dummy features
def create_dummy_vars(df):
# Create dummy variables for all 'object' type variables except 'Loan_Status'
    #df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    
    # store the processed dataset in data/processed
    #df.to_csv('./src/data/processed/Processed_Credit_Dataset.csv', index=None)

    # Separate the input features and target variable
    #X = df.drop('price', axis=1)
    #y = df['price']

    return df