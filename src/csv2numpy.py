import pandas as pd
import numpy as np


def csv2np(data):

    df = pd.read_csv(data)
    df["Date"] = pd.to_datetime(df["Date"])

    df_np = df[df["Location"] == "Uluru"].to_numpy()

    idx_no = np.where(df_np == 'No')
    idx_yes = np.where(df_np == 'Yes')
    idx_nan = np.where(df_np == 'nan')

    df_np[idx_no] = False
    df_np[idx_yes] = True
    df_np[idx_nan] = np.nan

    return df_np


if __name__ == "__main__":
    data = "../data/weatherAUS.csv"

    weather_np = csv2np(data)
    print(weather_np)

