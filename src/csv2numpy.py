import pandas as pd


def csv2np(data):

    df = pd.read_csv(data)
    df['Date'] = pd.to_datetime(df['Date'])
    print(df[ df['Location']=='Uluru'] )

    return df.to_numpy()


if __name__ == "__main__":
    data = "./data/weatherAUS.csv"

    weather_np = csv2np(data)
    print( weather_np[:,1] )
    print( weather_np.shape )
