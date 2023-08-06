import pandas as pd
import numpy as np
import csv

df_beforeT = pd.read_csv('C:/Users/logik/Desktop/Health Gym adapting ver/Demo/A000_Inputs/MIMICdata2019.csv')
rescale_N = ['age',  'elixhauser','Weight_kg', 'HR', 'SysBP',	'MeanBP',	'DiaBP',	'RR',	'Potassium',	'Sodium',
             'Calcium',		'Albumin',   'Hb', 'Arterial_pH',	'Arterial_BE',	'HCO3',
             'Shock_Index', 'cumulated_balance'
           ]

rescale_LN = [ "BUN",          "Creatinine",
    "Total_bili",    "WBC_count",     "Platelets_count",
    "PaO2_FiO2",  "paCO2",        "Arterial_lactate",
    "input_total",  "output_total",  "output_4hourly"
    ]

def get_minmax(x):
    return [np.log(min(df_beforeT[x]) + 1), np.log(max(df_beforeT[x]) + 1)]

with open('min_max.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Item', 'Min', 'Max'])

    for i in rescale_LN:
        minmax = get_minmax(i)
        writer.writerow([i] + minmax)

    for i in rescale_N:
        minmax = [min(df_beforeT[i]), max(df_beforeT[i])]
        writer.writerow([i] + minmax)

print('Results have been written to min_max.csv.')
