import pandas as pd


def csv2np(data):

    df = pd.read_csv(data)

    return df.to_numpy()


if __name__ == "__main__":
    data = "./data/weatherAUS.csv"

    weather_np = csv2np(data)
