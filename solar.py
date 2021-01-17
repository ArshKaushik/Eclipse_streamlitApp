import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

solar = pd.read_csv('SolarEclipse.csv')

solar_features = solar.drop(['Saros Number', 'Eclipse Type', 'Gamma'], axis = 1)
solar_labels = solar[['Saros Number', 'Eclipse Type', 'Gamma']].copy()

solar_model = RandomForestRegressor()
solar_model.fit(solar_features, solar_labels)

joblib.dump(solar_model, 'solar_joblib')