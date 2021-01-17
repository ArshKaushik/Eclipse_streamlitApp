import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

@st.cache
def get_data(fileName):
    dataFrame = pd.read_csv(fileName)
    return dataFrame

lunar = get_data('LunarEclipse.csv')
lunar_features = lunar.drop(['Saros Number', 'Eclipse Type', 'Gamma'], axis = 1)
lunar_labels = lunar[['Saros Number', 'Eclipse Type', 'Gamma']].copy()
lunar_corr = lunar.corr()
l_sn = lunar_corr['Saros Number'].sort_values(ascending=False)
l_et = lunar_corr['Eclipse Type'].sort_values(ascending=False)
l_gm = lunar_corr['Gamma'].sort_values(ascending=False)
lunar_model = joblib.load('lunar_joblib')

solar = get_data('SolarEclipse.csv')
solar_features = solar.drop(['Saros Number', 'Eclipse Type', 'Gamma'], axis = 1)
solar_labels = solar[['Saros Number', 'Eclipse Type', 'Gamma']].copy()
solar_corr = solar.corr()
s_sn = solar_corr['Saros Number'].sort_values(ascending=False)
s_et = solar_corr['Eclipse Type'].sort_values(ascending=False)
s_gm = solar_corr['Gamma'].sort_values(ascending=False)
solar_model = joblib.load('solar_joblib')

# TODO: data visualization for lunar and solar inside the app

st.sidebar.image('Asset 4.png', width=100)
st.sidebar.title('Navigate')
model_selection = ['None', 'Lunar', 'Solar']
model = st.sidebar.selectbox('Select your model', model_selection)

if model == 'None':
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(1, 100, 5):
        latest_iteration.text(f'Loading {i+4}%')
        bar.progress(i)
        time.sleep(0.05)
    bar.progress(100)
    st.success('Application loaded successfully')
    st.balloons()
    st.title('Welcome to ECLIPSE!')
    st.markdown('ECLIPSE, an ML model, helps to visualize the key parameters of an eclipse. It takes parameters related to eclipse from the user to process it as per the pre-trained data using random forest regressor algorithm to produce near precise predictions, as any ML model is not perfectly accurate. This model does away with the immense calculations which need to be carried out by the researchers and scientists in order to predict the eclipse and yields instant results which will save the time and complexity of the calculations that the researcher might have to face.')
    st.image('Eclipse.png', width=700)
    st.write('---')

# -------------------- LUNAR ECLIPSE --------------------
if model == 'Lunar':
    st.title('Lunar Eclipse')
    st.header('Dataset')
    st.write(lunar[:100])
    st.info('Because of large size of the dataset, only first 100 rows have been shown')

    if st.checkbox('Info'):
        st.markdown('* Latitudes of the southern hemisphere have been denoted by a negative sign whereas the latitides of northern hemisphere have been denoted by a positive sign')
        st.markdown('* Longitudes of the western hemisphere have been denoted by a negative sign whereas the longitudes of the eastern hemisphere have been denoted by a positive sign')
        st.markdown('* Some change have been made to eclipse type column. Check below:')
        eclipseTypeChanges = {
            'Value assigned': [1, 2, 3, 4, 5, 6, 7, 8],
            'Eclipse type': ['N', 'Nb', 'Ne', 'Nx', 'P', 'T', 'T-', 'T+']
        }
        l_etc = pd.DataFrame(eclipseTypeChanges)
        st.write(l_etc)

    if st.checkbox('Correlation'):
        st.subheader('Correlation with saros number:\n')
        st.write(l_sn)
        st.subheader('Correlation with eclipse type:\n')
        st.write(l_et)
        st.subheader('Correlation with gamma:\n')
        st.write(l_gm)

    st.write('---')
    st.header('Predictions')
    st.sidebar.header('Input parameters')

    year = st.sidebar.number_input('Year', value = 2020)
    month = st.sidebar.slider('Month', 1, 12, 1)
    day = st.sidebar.slider('Day', 1, 31, 10)
    hours = st.sidebar.slider('Hours', 1, 24, 10)
    minutes = st.sidebar.slider('Minutes', 1, 60, 10)
    seconds = st.sidebar.slider('Seconds', 1, 60, 10)
    delta_T = st.sidebar.number_input('Delta T (s)', value = 100, step = 1)
    lunation_number = st.sidebar.number_input('Lunation number (int)', value = 0, step = 1)
    penumbral_magnitude = st.sidebar.number_input('Penumbral magnitude (+ve)')
    umbral_magnitude = st.sidebar.number_input('Umbral magnitude (-ve accepted)')
    latitude = st.sidebar.slider('Latitude', int(lunar['Latitude'].min()), int(lunar['Latitude'].max()))
    longitude = st.sidebar.slider('Longitude', int(lunar['Longitude'].min()), int(lunar['Longitude'].max()))
    penumbral_eclipse_duration = st.sidebar.number_input('Penumbral eclipse duration (m)')

    st.subheader('Specified input parameters')
    data = {
        'Year': year,
        'Month': month,
        'Day': day,
        'Hours': hours,
        'Minutes': minutes,
        'Seconds': seconds,
        'Delta T (s)': delta_T,
        'Lunation Number': lunation_number,
        'Penumbral Magnitude': penumbral_magnitude,
        'Umbral Magnitude': umbral_magnitude,
        'Latitude': latitude,
        'Longtitude': longitude,
        'Penumbra Eclipse Duration (m)': penumbral_eclipse_duration
    }
    df = pd.DataFrame(data, index = [0])
    st.write(df)

    features = np.array([[
                year, month, day, hours, minutes, seconds, delta_T, lunation_number, penumbral_magnitude,
                umbral_magnitude, latitude, longitude, penumbral_eclipse_duration
        ]])
    predictions = lunar_model.predict(features)

    st.subheader('Saros Number')
    st.write(int(predictions[0][0]))

    eclipseDict = {
        1 : 'N', 2 : 'Nb', 3 : 'Ne', 4 : 'Nx', 5 : 'P', 6 : 'T', 7 : 'T-', 8 : 'T+'
    }
    n = int(predictions[0][1])
    val = ''
    for i in eclipseDict.keys():
        if n == i:
            val = eclipseDict[i]
    st.subheader('Eclipse Type') 
    st.write(val)
    
    st.subheader('Gamma')
    st.write(round(predictions[0][2], 4))
    st.text('')

    if st.checkbox('Cross validation score'):

        scores = cross_val_score(lunar_model, lunar_features, lunar_labels, scoring = 'neg_mean_squared_error', cv = 10)
        rmse_scores = np.sqrt(-scores)

        aa, bb = st.beta_columns(2)

        aa.subheader('Root mean squared error')
        aa.write(rmse_scores)

        bb.subheader('Mean')
        bb.write(rmse_scores.mean())

        bb.subheader('Standard Deviation')
        bb.write(rmse_scores.std())

    st.write('---')
    st.header('Data visualization')
    st.subheader('Line chart')
    st.line_chart(lunar)
    st.subheader('Area chart')
    st.area_chart(lunar)

# -------------------- SOLAR ECLIPSE --------------------
if model == 'Solar':
    st.title('Solar Eclipse')
    st.header('Dataset')
    st.write(solar[:100])
    st.info('Because of large size of the dataset, only first 100 rows have been shown')

    if st.checkbox('Info'):
        st.markdown('* Latitudes of the southern hemisphere have been denoted by a negative sign whereas the latitides of northern hemisphere have been denoted by a positive sign')
        st.markdown('* Longitudes of the western hemisphere have been denoted by a negative sign whereas the longitudes of the eastern hemisphere have been denoted by a positive sign')
        st.markdown('* Some change have been made to eclipse type column. Check below:')
        eclipseTypeChanges = {
            'Value assigned': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            'Eclipse type': ['A', 'A-', 'A+', 'Am', 'An', 'As', 'H', 'H2', 'H3', 'Hm', 'P', 'Pb', 'Pe', 'T', 'T-', 'T+', 'Tm', 'Tn', 'Ts']
        }
        s_etc = pd.DataFrame(eclipseTypeChanges)
        st.write(s_etc)

    if st.checkbox('Correlation'):
        st.subheader('Correlation with saros number:\n')
        st.write(s_sn)
        st.subheader('Correlation with eclipse type:\n')
        st.write(s_et)
        st.subheader('Correlation with gamma:\n')
        st.write(s_gm)

    st.write('---')
    st.header('Predictions')
    st.sidebar.header('Input parameters')

    year = st.sidebar.number_input('Year', value = 2020)
    month = st.sidebar.slider('Month', 1, 12, 1)
    day = st.sidebar.slider('Day', 1, 31, 10)
    hours = st.sidebar.slider('Hours', 1, 24, 10)
    minutes = st.sidebar.slider('Minutes', 1, 60, 10)
    seconds = st.sidebar.slider('Seconds', 1, 60, 10)
    delta_T = st.sidebar.number_input('Delta T (s)', value = 100, step = 1)
    lunation_number = st.sidebar.number_input('Lunation number (int)', value = 0, step = 1)
    eclipse_magnitude = st.sidebar.number_input('Eclipse Magnitude', value = 1.0617)
    latitude = st.sidebar.slider('Latitude', int(lunar['Latitude'].min()), int(lunar['Latitude'].max()))
    longitude = st.sidebar.slider('Longitude', int(lunar['Longitude'].min()), int(lunar['Longitude'].max()))
    sun_altitude = st.sidebar.slider('Sun Altitude', int(solar['Sun Altitude'].min()), int(solar['Sun Altitude'].max()))
    sun_azimuth = st.sidebar.slider('Sun Azimuth', int(solar['Sun Azimuth'].min()), int(solar['Sun Azimuth'].max()))

    st.subheader('Specified input parameters')
    data = {
        'Year': year,
        'Month': month,
        'Day': day,
        'Hours': hours,
        'Minutes': minutes,
        'Seconds': seconds,
        'Delta T (s)': delta_T,
        'Lunation Number': lunation_number,
        'Eclipse Magnitude': eclipse_magnitude,
        'Latitude': latitude,
        'Longtitude': longitude,
        'Sun Altitude': sun_altitude,
        'Sun Azimuth': sun_azimuth
    }
    df = pd.DataFrame(data, index = [0])
    st.write(df)

    features = np.array([[
                year, month, day, hours, minutes, seconds, delta_T, 
                lunation_number, eclipse_magnitude, 
                latitude, longitude, sun_altitude, sun_azimuth
        ]])
    predictions = solar_model.predict(features)

    st.subheader('Saros Number')
    st.write(int(predictions[0][0]))

    eclipseDict = {
        1 : 'A', 2 : 'A-', 3 : 'A+', 4 : 'Am', 5 : 'An', 6 : 'As', 7 : 'H', 8 : 'H2', 9 : 'H3', 10 : 'Hm',
        11 : 'P', 12 : 'Pb', 13 : 'Pe', 14 : 'T', 15 : 'T-', 16 : 'T+', 17 : 'Tm', 18 : 'Tn', 19 : 'Ts'
    }
    n = int(predictions[0][1])
    val = ''
    for i in eclipseDict.keys():
        if n == i:
            val = eclipseDict[i]
    st.subheader('Eclipse Type') 
    st.write(val)
    
    st.subheader('Gamma')
    st.write(round(predictions[0][2], 4))
    st.text('')

    if st.checkbox('Cross validation score'):

        scores = cross_val_score(solar_model, solar_features, solar_labels, scoring = 'neg_mean_squared_error', cv = 10)
        rmse_scores = np.sqrt(-scores)

        aa, bb = st.beta_columns(2)

        aa.subheader('Root mean squared error')
        aa.write(rmse_scores)

        bb.subheader('Mean')
        bb.write(rmse_scores.mean())

        bb.subheader('Standard Deviation')
        bb.write(rmse_scores.std())

    st.write('---')
    st.header('Data visualization')
    st.subheader('Line chart')
    st.line_chart(solar)
    st.subheader('Area chart')
    st.area_chart(solar)
