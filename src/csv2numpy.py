import numpy as np
import pandas as pd
from astropy.time import Time
from geopy.geocoders import Nominatim


def raw_parse(data):
    data_list = []
    with open(data, "r") as f:
        for line in f.readlines():
            dat = line[:-1].split(",")
            if "NA" not in dat:
                data_list.append(dat)

    df_np = np.array(data_list)

    idx_no = np.where(df_np == "No")
    idx_yes = np.where(df_np == "Yes")

    df_np[idx_no] = False
    df_np[idx_yes] = True

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
    print(df_np.shape)
    print(
        Time(df_np[1:, 0], format="isot").gps
        - np.min(Time(df_np[1:, 0], format="isot").gps)
    )
    return


def geo_location(name):

    locator = Nominatim(user_agent="telescope-loc")
    location = locator.geocode(name)

    lat = location.latitude

    lon = location.longitude

    return lat, lon


if __name__ == "__main__":
    data = "../data/weatherAUS.csv"
    raw_parse(data)

#     weather_np = csv2np(data)

    lat, lon = geo_location("Melbourne")
    print(lat, lon)
