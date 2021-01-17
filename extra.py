import numpy as np
import joblib

lunar_model = joblib.load('lunar_joblib')

year = 2020
month = 8
day = 21
hours = 21
minutes = 8
seconds = 3
delta = 50189
lunation_number = 1500
penumbral_magnitude = 3.778
umbral_magnitude = -1.8479
latitude = -21
longitude = 12
penumbral_eclipse_duration = 218

features = np.array([[
            year, month, day, hours, minutes, seconds, delta, 
            lunation_number, penumbral_magnitude, umbral_magnitude, 
            latitude, longitude, penumbral_eclipse_duration
    ]])

custom_value_predictions = lunar_model.predict(features)

def saros_number():
    print(f'Saros Number: {int(custom_value_predictions[0][0])}')

def eclipse_type():
    eclipseDict = {
        1 : 'N',
        2 : 'Nb',
        3 : 'Ne',
        4 : 'Nx',
        5 : 'P',
        6 : 'T',
        7 : 'T-',
        8 : 'T+'
    }
    n = int(custom_value_predictions[0][1])
    val = ''
    for i in eclipseDict.keys():
        if n == i:
            val = eclipseDict[i]
    print(f'Eclipse Type: {val}')

def gamma():    
    print(f'Gamma: {round(custom_value_predictions[0][2], 4)}')

saros_number()
eclipse_type()
gamma()
