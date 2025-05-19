from datetime import datetime
import yfinance as yf
import pandas as pd
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt



def get_args():
    parser = argparse.ArgumentParser(description="Stock Prediction")
    parser.add_argument("--root", '-r', type=str, default="data", help="Path to your data storage")
    parser.add_argument("--ticker", '-t', type=str, default="NVDA", help="Default: NVDA")
    parser.add_argument("--start_date", '-s', type=str, default="2020-01-01", help="Start_date download")
    parser.add_argument("--end_date", '-e', type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End_date download")
    args =parser.parse_args()
    return args

def download_data(args):

    if not os.path.exists(args.root):
        os.makedirs(args.root)
        print(f"Created directory: {args.root}")

    # Set end date to current date if not provided
    if args.end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Create filename based on ticker and date range
    filename = f"{args.ticker}_{args.start_date}_{args.end_date}.csv"
    file_path = os.path.join(args.root, filename)

    # Check if file already exists
    if os.path.exists(file_path):
        print(f"File {filename} already exists. Loading from file...")
        return pd.read_csv(file_path)

    # Download data since the file doesn't exist
    print(f"Downloading data for {args.ticker} from {args.start_date} to {args.end_date}...")
    data = yf.download(args.ticker, start=args.start_date, end=args.end_date)

    # Save to CSV
    if not data.empty:
        # Reset index to make Date a column
        data = data.reset_index()
        # Remove the first row
        data = data.iloc[1:].reset_index(drop=True)
        data.to_csv(file_path, index=False) # This parameter tells pandas NOT to include the DataFrame's index as a column in the CSV file
        print(f"Data saved to {file_path}")
    else:
        print(f"No data found for {args.ticker}")

    return data



# Focusing predict Close Price
def train(data):
    df = pd.DataFrame(data)
    df = df.dropna()
    # Drop duplicate columns to prevent bugs
    df = df.loc[:, ~df.columns.duplicated()]

    x = df.drop(["Date", "Close"], axis=1).astype(float)
    y = df["Close"].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    # Initialize
    num_tranf = Pipeline(steps=[
        ("Impute", SimpleImputer(strategy='median')),
        ("Scalar", StandardScaler())
    ])

    # Process data
    preprocessor = ColumnTransformer(transformers=[
        ("num_features", num_tranf, ["Open", "High", "Low", "Volume"])
    ])

    # Model
    reg = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])

    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    return y_test, y_pred

    # Result
    # for i, j in zip(y_test, y_pred):
    #     print("Actual value: {} --- Predict value: {}".format(i, j))

    #-------------------------------------------------------

    # Result:  {'preprocessor__num_features__Impute__strategy': 'median'} : 0.9997309120215596
    # # Test parameters

    # params = {"preprocessor__num_features__Impute__strategy":["median", "most_frequent"]}
    # model_reg = GridSearchCV(reg, param_grid=params, scoring="r2", cv=6, verbose=2, n_jobs=10)
    # model_reg.fit(x_train, y_train)
    # print(model_reg.best_score_)
    # print(model_reg.best_params_)

    # -------------------------------------------------------
def evaluate(y_test, y_pred, args):
    # Evaluate
    print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
    print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
    print("R2_Score: {}".format(r2_score(y_test, y_pred)))

    # Visualization
    ticker = args.ticker
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Predicted vs Actual")  # Add label here
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',
             label="Ideal Line")  # Ideal line for reference
    plt.title(f"Actual vs Predicted {ticker}  Price (Scatter)")
    plt.xlabel(f"Actual {ticker} Price")
    plt.ylabel(f"Predicted {ticker} Price")
    plt.grid(True)
    plt.legend()  # Show the legend
    plt.show()


if __name__ == "__main__":
    # Example usage
    args = get_args()
    data = download_data(args)
    y_test, y_pred = train(data)
    evaluate(y_test, y_pred, args)