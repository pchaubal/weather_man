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

    df_np[idx_no] = 0
    df_np[idx_yes] = 1

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

    # Replace  location name by lat long
#     print( geo_location('Alice Springs') )
    name = list(np.unique( df_np[1:,1]))
    name2 = [n.replace('MelbourneAirport','Melbourne') for n in name]
    name2 = [n.replace('AliceSprings','Alice Springs') for n in name2]
    name2 = [n.replace('PerthAirport','Perth') for n in name2]
    name2 = [n.replace('CoffsHarbour','Coffs Harbour') for n in name2]
    name2 = [n.replace('SydneyAirport','Sydney') for n in name2]
    name2 = [n.replace('NorfolkIsland','Norfolk Island') for n in name2]
    name2 = [n.replace('MountGambier','Mount Gambier') for n in name2]
    name2 = [n.replace('WaggaWagga','Wagga Wagga') for n in name2]
   

    latlon =np.asarray([geo_location(i) for i in name2])
    latlon_dict = {}
    for i,n in enumerate(name):
        latlon_dict[n] = latlon[i] 

    lat,lon = latlon[:,0], latlon[:,1]
    lat2 = np.zeros(len(df_np)-1)
    lon2 = np.zeros(len(df_np)-1)
    for n in name:
        idx= np.where(df_np[1:,1]==n)
        lat2[idx] = latlon_dict[n][0] 
        lon2[idx] = latlon_dict[n][1] 

    print('lat2', lat2 )
    print('lon2',  lon2 )



    #converts data stream inot floats
    gps_time = list(
        Time(df_np[1:, 0], format="isot").gps
        - np.min(Time(df_np[1:, 0], format="isot").gps)
    )
    gps_time =  gps_time

    month =  [int(i.split("-")[1]) for i in df_np[1:, 0]]

    data = np.c_[np.array(gps_time), np.array(month), lat2, lon2, df_np[1:, 2:]]
    print( data[:5] )

    np.savetxt('refined_data.txt', data.astype('float64'))
    


    return


def geo_location(name):

    locator = Nominatim(user_agent="telescope-loc")
    location = locator.geocode(name)
    
    try:
        lat = location.latitude

        lon = location.longitude
    except:
        print('Failed at:', name )

    return [lat, lon]


if __name__ == "__main__":
    data = "../data/weatherAUS.csv"
    raw_parse(data)

    #     weather_np = csv2np(data)

    #  lat, lon = geo_location("Melbourne")
