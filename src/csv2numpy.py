import numpy as np
import pandas as pd


def csv2np(data):

    df = pd.read_csv(data)
    df["Date"] = pd.to_datetime(df["Date"])

    df_np = df[df["Location"] == "Uluru"].to_numpy()

    idx_no = np.where(df_np == "No")
    idx_yes = np.where(df_np == "Yes")
    idx_nan = np.where(df_np == "nan")

    df_np[idx_no] = False
    df_np[idx_yes] = True
    df_np[idx_nan] = np.nan

    cardinal = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]

    for i, c in enumerate(cardinal):
        idx_c = np.where(df_np == c)
        df_np[idx_c] = i + 1

    return df_np


if __name__ == "__main__":
    data = "../data/weatherAUS.csv"

    weather_np = csv2np(data)
    print(
        weather_np[0]
    )
