import pandas as pd
import math
df = pd.read_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/predictions_100.csv')

df['Demo002_Age'] = df['Demo002_Age'] * (33383.85 - 6582.4222337963) + 6582.4222337963
df['Demo004_Elixhauser score'] = df['Demo004_Elixhauser score'] * (14 - 0) + 0
df['Demo005_Weight_kg'] = df['Demo005_Weight_kg'] * (278.5 - 0) + 0
df['Vitl002_HR'] = df['Vitl002_HR'] * (209 - 0) + 0
# df['Vitl012_HRmin'] = df['Vitl012_HRmin'] * (209 - 0) + 0
# df['Vitl013_HRmax'] = df['Vitl013_HRmax'] * (241 - 0) + 0
df['Vitl003_SysBP'] = df['Vitl003_SysBP'] * (270 - 0) + 0
# df['Vitl014_SysBPmin'] = df['Vitl014_SysBPmin'] * (270 - 0) + 0
# df['Vitl015_SysBPmax'] = df['Vitl015_SysBPmax'] * (287 - 0) + 0
df['Vitl004_MeanBP'] = df['Vitl004_MeanBP'] * (200 - 0) + 0
# df['Vitl016_MeanBPmin'] = df['Vitl016_MeanBPmin'] * (200 - 0) + 0
# df['Vitl017_MeanBPmax'] = df['Vitl017_MeanBPmax'] * (205.333333333333 - 0) + 0
df['Vitl005_DiaBP'] = df['Vitl005_DiaBP'] * (247 - 0) + 0
# df['Vitl018_DiaBPmin'] = df['Vitl018_DiaBPmin'] * (247 - 0) + 0
# df['Vitl019_DiaBPmax'] = df['Vitl019_DiaBPmax'] * (247 - 0) + 0
df['Vitl006_RR'] = df['Vitl006_RR'] * (77 - 0) + 0
# df['Vitl020_RRmin'] = df['Vitl020_RRmin'] * (77 - 0) + 0
# df['Vitl021_RRmax'] = df['Vitl021_RRmax'] * (80 - 0) + 0
df['Labs001_Potassium'] = df['Labs001_Potassium'] * (12.7 - 1.5) + 1.5
df['Labs002_sodium'] = df['Labs002_sodium'] * (177 - 95) + 95
# df['Labs003_chloride'] = df['Labs003_chloride'] * (150 - 70) + 70
df['Labs008_Ca'] = df['Labs008_Ca'] * (19.2 - 0) + 0
# df['Labs009_IonisedCa'] = df['Labs009_IonisedCa'] * (4.3 - 0) + 0
# df['Labs010_CO2'] = df['Labs010_CO2'] * (112 - 0) + 0
df['Labs014_Albumin'] = df['Labs014_Albumin'] * (5.9 - 1) + 1
df['Labs015_Hb'] = df['Labs015_Hb'] * (21.9720475192173 + 0.424877707896576) - 0.424877707896576
df['Labs021_pH'] = df['Labs021_pH'] * (7.76 - 6.73) + 6.73
df['Labs024_BE'] = df['Labs024_BE'] * (30 + 30) - 30
df['Labs025_HCO3'] = df['Labs025_HCO3'] * (60 - 0) + 0
# df['Vent002_FiO2'] = df['Vent002_FiO2'] * (1 - 0.2) + 0.2
df['Vitl011_Shock_Index'] = df['Vitl011_Shock_Index'] * (2 + 1.31884057971015) - 1.31884057971015
df['Flud006_Cumulated fluid balance'] = df['Flud006_Cumulated fluid balance'] * (98717.8217752156 + 99010) - 99010
## log-normal group
# df['Labs004_Glucose'] = math.e ** (df['Labs004_Glucose'] * (6.90173720665657 - 0.693147180559945) + 0.693147180559945) - 1
df['Labs005_BUN'] = math.e ** (df['Labs005_BUN'] * (6.49223983502047 - 0) + 0) - 1
df['Labs006_Creatinine'] = math.e ** (df['Labs006_Creatinine'] * (4.997212274 - 0) + 0) - 1
# df['Labs007_Mg'] = math.e ** (df['Labs007_Mg'] * (2.360854001 - 0) + 0) - 1
# df['Labs011_SGOT'] = math.e ** (df['Labs011_SGOT'] * (9.2043223 - 0) + 0) - 1
# df['Labs012_SGPT'] = math.e ** (df['Labs012_SGPT'] * (9.193295937 - 0) + 0) - 1
df['Labs013_TotalBili'] = math.e ** (df['Labs013_TotalBili'] * (4.53387647 - 0) + 0) - 1
df['Labs016_WbcCount'] = math.e ** (df['Labs016_WbcCount'] * (6.139022111 - 0) + 0) - 1
df['Labs017_PlateletsCount'] = math.e ** (df['Labs017_PlateletsCount'] * (7.591357047 - 1.791759469) + 1.791759469) - 1
# df['Labs018_PTT'] = math.e ** (df['Labs018_PTT'] * (5.204006687 - 0) + 0) - 1
# df['Labs019_PT'] = math.e ** (df['Labs019_PT'] * (5.017279837 - 2.140066163) + 2.140066163) - 1
# df['Labs022_PaO2'] = math.e ** (df['Labs022_PaO2'] * (6.533788838 - 0) + 0) - 1
df['Labs027_PaO2_FiO2 ratio'] = math.e ** (df['Labs027_PaO2_FiO2 ratio'] * (7.969522443 - 0) + 0) - 1
df['Labs023_PaCO2'] = math.e ** (df['Labs023_PaCO2'] * (5.272999559 - 0) + 0) - 1
df['Labs026_Lactate'] = math.e ** (df['Labs026_Lactate'] * (3.411147713 - 0) + 0) - 1
df['Flud001_InputTotal'] = math.e ** (df['Flud001_InputTotal'] * (11.51293546 - 0) + 0) - 1
# df['Flud002_Input4H'] = math.e ** (df['Flud002_Input4H'] * (9.756885797 - 0) + 0) - 1
# df['Flud003_MaxVaso'] = math.e ** (df['Flud003_MaxVaso'] * (3.044522438 - 0) + 0) - 1
# df['Flud008_MedianVaso'] = math.e ** (df['Flud008_MedianVaso'] * (3.044522438 - 0) + 0) - 1
df['Flud004_OutputTotal'] = math.e ** (df['Flud004_OutputTotal'] * (11.51293546 - 0) + 0) - 1
df['Flud005_Output4H'] = math.e ** (df['Flud005_Output4H'] * (8.517393171 - 0) + 0) - 1

# df = df.drop('Unnamed: 0', axis=1)
df.to_csv('C:/Users/logik/Desktop/rlsepsis234-master - full/predictions_100_descale.csv')










