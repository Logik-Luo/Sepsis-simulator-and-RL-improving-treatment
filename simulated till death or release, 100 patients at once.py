import joblib
import csv

import numpy as np
import pandas as pd
import tensorflow as tf

model_for_y = joblib.load('C:/Users/logik/Desktop/rlsepsis234-master - full/lightgbm_model.pkl')
model_for_X = tf.keras.models.load_model(
    "C:/Users/logik/Desktop/rlsepsis234-master - full/gym_sepsis/env/model/sepsis_states.model")  # State model

df = pd.read_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/data/test_state_action_reward_df.csv')

# List of new column names
column_names = ['Vitl009_SOFA', 'SpO2', 'Temp_C', 'GCS', 'Demo002_Age', 'Demo004_Elixhauser score', 'Demo005_Weight_kg',
                'Vitl002_HR', 'Vitl003_SysBP', 'Vitl004_MeanBP', 'Vitl005_DiaBP', 'Vitl006_RR', 'Labs001_Potassium',
                'Labs002_sodium', 'Labs008_Ca', 'Labs014_Albumin', 'Labs015_Hb', 'Labs021_pH', 'Labs024_BE',
                'Labs025_HCO3', 'Vitl011_Shock_Index', 'Flud006_Cumulated fluid balance', 'Labs005_BUN', 'Labs006_Creatinine',
                'Labs013_TotalBili', 'Labs016_WbcCount', 'Labs017_PlateletsCount', 'Labs027_PaO2_FiO2 ratio', 'Labs023_PaCO2',
                'Labs026_Lactate', 'Flud001_InputTotal', 'Flud004_OutputTotal', 'Flud005_Output4H', 'Demo001_Gender_0',
                'Demo001_Gender_1', 'Vent003_sedation_0', 'Vent003_sedation_1', 'Flud007_rrt_0', 'Flud007_rrt_1',
                'Vent001_Mech_0', 'Vent001_Mech_1', 'Vitl010_SIRS_1', 'Vitl010_SIRS_2', 'Vitl010_SIRS_3', 'Vitl010_SIRS_4',
                'Vitl010_SIRS_5', 'vaso_input', 'iv_input', 'discrete_action']

# Open the csv file in write mode
with open('C:/Users/logik/Desktop/rlsepsis234-master - full/predictions_100.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(['Admn001_ID', 'bloc'] + column_names)

    for patient_id in range(100):  # 100 patients
        while True:
            start = np.random.randint(0, len(df) - 10)  # returns a random integer from low (inclusive) to high (exclusive)
            window = df.iloc[start:start + 10]  # then takes that integer (inclusive) and 9 behind it

            # Check if all values in 'Demo002_Age' are the same
            if window['Demo002_Age'].nunique() == 1:
                # If they are, convert the DataFrame to a numpy array and break the loop
                window = window.values
                break

        window = window[:, 3:52] # select columns for features

        # Define a counter for labels 0
        counter_0 = 0

        while True:
            window_reshaped = window.reshape(1, 10, 49)  # 1 row, 10 steps, 49 columns

            # Generate label using model_for_y
            y = model_for_y.predict(window)

            # Generate features using model_for_X
            X = model_for_X.predict(window_reshaped)

            # Append the new features as the last row, and remove the first row
            window = np.vstack([window[1:], X])  # returns vertically stacked array

            # If the predicted label is 0, increase the counter_0
            if y[0] == 0:
                counter_0 += 1
                print(f'Patient {patient_id} - Label {y[0]} was predicted, encountered {counter_0} times.')

            # If we've hit the prediction limit, change the label to 3
            if counter_0 == 80:
                y[0] = 3
                print(f'Patient {patient_id} - Label 0 was predicted 80 times, changing the label to 1 and stopping the loop.')

            # Write to the csv file
            writer.writerow([patient_id] + [int(y[0])] + X[0].tolist())

            # If the predicted label is 1 or 2, or we've hit the prediction limit, stop the loop
            if y[0] == 1 or y[0] == 2 or counter_0 == 80:
                print(f'Patient {patient_id} - Label {y[0]} was predicted, stopping the loop.')
                break
