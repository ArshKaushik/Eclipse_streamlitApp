import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

lunar = pd.read_csv('LunarEclipse.csv')

lunar_features = lunar.drop(['Saros Number', 'Eclipse Type', 'Gamma'], axis = 1)
lunar_labels = lunar[['Saros Number', 'Eclipse Type', 'Gamma']].copy()

lunar_model = RandomForestRegressor()
lunar_model.fit(lunar_features, lunar_labels)

joblib.dump(lunar_model, 'lunar_joblib')