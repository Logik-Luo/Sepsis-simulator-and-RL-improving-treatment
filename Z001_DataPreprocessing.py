###===###
# Import some dependencies
import  numpy                   as      np
import  pandas                  as      pd
from    sklearn.preprocessing   import  MinMaxScaler
import  matplotlib.pyplot       as      plt

import 	torch

###===###
# Download our synthetic sepsis dataset from PhysioNet
# and place it in the A000_Inputs folder
Folder  = "C:/Users/logik/Desktop/rlsepsis234-master - full/data/"
File    = "MIMICdata2019.csv"

###===###
# Read in the synthetic sepsis dataset and treat it as the ground truth
MyData = pd.read_csv(Folder+File)

# Drop the unneeded columns
MyData = MyData.drop(["delay_end_of_record_and_discharge_or_death", "charttime", "dayweek", "died_within_48h_of_out_time", 'sepsis_onset'], axis = 1)

MyData['Unnamed: 0'] = np.arange(0, len(MyData))
# Rename the columns for our sanity
# 	Admn: 	Administrative purposes
# 	Demo: 	Demographics
# 	Vitl: 	Vital signs/variables
# 	Labs: 	Lab results
# 	Flud: 	Fluid measurements
# 	Vent: 	Ventilation
ColNameSwap = {"icustayid":         "Admn001_ID",
               "bloc":              "bloc",
               "gender":            "Demo001_Gender",
               # "re_admission":      "Demo003_ReAd",
               "age":               "Demo002_Age",
               "elixhauser":        "Demo004_Elixhauser score",
               "Weight_kg":         "Demo005_Weight_kg",
               # "GCS":               "Vitl001_GCS",
               "HR":                "Vitl002_HR",
               "SysBP":             "Vitl003_SysBP",
               "MeanBP":            "Vitl004_MeanBP",
               "DiaBP":             "Vitl005_DiaBP",
               "RR":                "Vitl006_RR",
               # "Temp_C":            "Vitl008_Temp",
               # "SpO2":              "Vitl007_SpO2",
               "SOFA":              "Vitl009_SOFA",
               "SIRS":              "Vitl010_SIRS",
               "Shock_Index":       "Vitl011_Shock_Index",
               # "HRmin":             "Vitl012_HRmin",
               # "HRmax":             "Vitl013_HRmax",
               # "SysBPmin":          "Vitl014_SysBPmin",
               # "SysBPmax":          "Vitl015_SysBPmax",
               # "MeanBPmin":         "Vitl016_MeanBPmin",
               # "MeanBPmax":         "Vitl017_MeanBPmax",
               # "DiaBPmin":          "Vitl018_DiaBPmin",
               # "DiaBPmax":          "Vitl019_DiaBPmax",
               # "RRmin":             "Vitl020_RRmin",
               # "RRmax":             "Vitl021_RRmax",
               # "HRsd":             "Vitl022_HRsd",
               # "RRsd":             "Vitl023_RRsd",
               # "MeanBPsd":         "Vitl024_MeanBPsd",
               # "SysBPsd":          "Vitl025_SysBPsd",
               # "DiaBPsd":          "Vitl026_DiaBPsd",

               "Potassium":         "Labs001_Potassium",
               "Sodium":            "Labs002_sodium",
               # "Chloride":          "Labs003_chloride",
               # "Glucose":           "Labs004_Glucose",
               "BUN":               "Labs005_BUN",
               "Creatinine":        "Labs006_Creatinine",
               # "Magnesium":         "Labs007_Mg",
               "Calcium":           "Labs008_Ca",
               # "Ionised_Ca":        "Labs009_IonisedCa",
               # "CO2_mEqL":          "Labs010_CO2",
               # "SGOT":              "Labs011_SGOT",
               # "SGPT":              "Labs012_SGPT",
               "Total_bili":        "Labs013_TotalBili",
               "Albumin":           "Labs014_Albumin",
               "Hb":                "Labs015_Hb",
               "WBC_count":         "Labs016_WbcCount",
               "Platelets_count":   "Labs017_PlateletsCount",
               # "PTT":               "Labs018_PTT",
               # "PT":                "Labs019_PT",
               # "INR":               "Labs020_INR",
               # "paO2":              "Labs022_PaO2",
               "paCO2":             "Labs023_PaCO2",
               "Arterial_lactate":  "Labs026_Lactate",
               "Arterial_pH":       "Labs021_pH",
               "Arterial_BE":       "Labs024_BE",
               "HCO3":              "Labs025_HCO3",
               "PaO2_FiO2":         "Labs027_PaO2_FiO2 ratio",

               "input_total":       "Flud001_InputTotal",
               "input_4hourly":     "Flud002_Input4H",
               "max_dose_vaso":     "Flud003_MaxVaso",
               # "median_dose_vaso":  "Flud008_MedianVaso",
               "output_total":      "Flud004_OutputTotal",
               "output_4hourly":    "Flud005_Output4H",
               "cumulated_balance": "Flud006_Cumulated fluid balance",
               "rrt":               "Flud007_rrt",

               # "FiO2_1":            "Vent002_FiO2",
               "mechvent":          "Vent001_Mech",
               "sedation":          "Vent003_sedation",
               
               "died_in_hosp":      "OutC001_hospital mortality",
               "mortality_90d":     "OutC002_90d mortality"

               }

# Perform name swapping
MyData.rename(
    columns = {**ColNameSwap, **{v:k for k,v in ColNameSwap.items()}},
    inplace=True)
###===###
# Create A001_DataTypes.csv to document data property
MyData_Types = pd.DataFrame()

# Including
# 	index: 		--
# 	name:  		--
# 	type:  		Real/binary/categorical
# 	num_classes:	The amount of levels for each variable; fixed 1 for real
# 	embedding_size:	Projection dimension using soft-embeddings
# 	index_start: 	The first variable location in the concatenated features
# 	index_end: 	The pairing last location
MyData_Types["index"]           = []	
MyData_Types["name"]            = []
MyData_Types["type"]            = []
MyData_Types["num_classes"]     = [] 	
MyData_Types["embedding_size"]  = []
MyData_Types["include"]         = []
MyData_Types["index_start"]     = []
MyData_Types["index_end"]       = []

###===###
# Create called A002_MyData.csv to store a machine-readable ground-truth dataset
MyData_Transformed = pd.DataFrame()

# No transformation required for patient ID
MyData_Transformed["Admn001_ID"] = MyData["Admn001_ID"]
MyData_Transformed["bloc"] = MyData["bloc"]
MyData_Transformed["Unnamed: 0"] = MyData["Unnamed: 0"]
MyData_Transformed["Vitl009_SOFA"] = MyData["Vitl009_SOFA"]     # gcs spo2 temp
MyData_Transformed["Flud002_Input4H"] = MyData["Flud002_Input4H"]
MyData_Transformed["Flud003_MaxVaso"] = MyData["Flud003_MaxVaso"]
MyData_Transformed["OutC002_90d mortality"] = MyData["OutC002_90d mortality"]
MyData_Transformed["timeday"] = MyData["timeday"]
MyData_Transformed["SpO2"] = MyData["SpO2"]
MyData_Transformed["Temp_C"] = MyData["Temp_C"]
MyData_Transformed["GCS"] = MyData["GCS"]

# Transformation procedure varies for
# 	Flt: float
# 	Bin: binary
# 	Cat: categorical

# There are 2 different types of flt variables
# 	N2: Those with Naturally Normal (N2) distributions
# 	LN: Those that can be Logged to become Normal (LN) 


# Flt_Variable_N2 = \
# [   "Demo002_Age",          "Demo004_Elixhauser score",         "Demo005_Weight_kg",
#     "Vitl002_HR",           "Vitl012_HRmin",        "Vitl013_HRmax",
#     "Vitl003_SysBP",        "Vitl014_SysBPmin",     "Vitl015_SysBPmax",
#     "Vitl004_MeanBP",       "Vitl016_MeanBPmin",    "Vitl017_MeanBPmax",
#     "Vitl005_DiaBP",        "Vitl018_DiaBPmin",     "Vitl019_DiaBPmax",
#     "Vitl006_RR",           "Vitl020_RRmin",        "Vitl021_RRmax",
#     "Labs001_Potassium",            "Labs002_sodium",           "Labs003_chloride",
#     "Labs008_Ca",           "Labs009_IonisedCa",    "Labs010_CO2",
#     "Labs014_Albumin",      "Labs015_Hb",           "Labs021_pH",
#     "Labs024_BE",           "Labs025_HCO3",
#     "Vent002_FiO2",         "Vitl011_Shock_Index",  "Flud006_Cumulated fluid balance"]

Flt_Variable_N2 = \
[   "Demo002_Age",          "Demo004_Elixhauser score",         "Demo005_Weight_kg",
    "Vitl002_HR",
    "Vitl003_SysBP",
    "Vitl004_MeanBP",
    "Vitl005_DiaBP",
    "Vitl006_RR",
    "Labs001_Potassium",            "Labs002_sodium",
    "Labs008_Ca",
    "Labs014_Albumin",      "Labs015_Hb",           "Labs021_pH",
    "Labs024_BE",           "Labs025_HCO3",
    "Vitl011_Shock_Index",  "Flud006_Cumulated fluid balance"]

# Flt_Variable_LN = \
# [   "Labs004_Glucose",      "Labs005_BUN",          "Labs006_Creatinine",
#     "Labs007_Mg",           "Labs011_SGOT",         "Labs012_SGPT",
#     "Labs013_TotalBili",    "Labs016_WbcCount",     "Labs017_PlateletsCount",   "Labs018_PTT", "Labs019_PT",
#     "Labs022_PaO2",         "Labs027_PaO2_FiO2 ratio",  "Labs023_PaCO2",        "Labs026_Lactate",
#     "Flud001_InputTotal",    "Flud008_MedianVaso",
#     "Flud004_OutputTotal",  "Flud005_Output4H",
#     ]

Flt_Variable_LN = \
[   "Labs005_BUN",          "Labs006_Creatinine",
    "Labs013_TotalBili",    "Labs016_WbcCount",     "Labs017_PlateletsCount",
    "Labs027_PaO2_FiO2 ratio",  "Labs023_PaCO2",        "Labs026_Lactate",
    "Flud001_InputTotal",
    "Flud004_OutputTotal",  "Flud005_Output4H",
    ]

#---
# Bin variables
# Bin_Variable = \
# [   "Demo001_Gender",      "Demo003_ReAd",      "Vent003_sedation",  "Flud007_rrt",
#     "Vent001_Mech",
#     ]

# Bin variables
Bin_Variable = \
[   "Demo001_Gender",       "Vent003_sedation",  "Flud007_rrt",
    "Vent001_Mech",
    ]

#---
# There are 2 different types of cat variables
# 	MTC: Those naturally with MulTi-Classes (MTC)
# 	NLN: Those flt-s that canNot be Logged to get Normal distribution (NLN)
# Cat_Variable_MTC = \
# ["Vitl001_GCS"]

Cat_Variable_SIRS = \
[   "Vitl010_SIRS"
    ]

# Cat_Variable_NLN = \
# [   "Vitl007_SpO2",         "Vitl008_Temp",      "Labs020_INR",      # sd were all dropped
#     ]

# Cat_Variable_NLN = \
# [   "Vitl007_SpO2",         "Vitl008_Temp",          # sd were all dropped
#     ]
# for sofa, gcs, spo2, temp: transform into 4 quartiles, nooooooooooooooooooooooo! not good for DR

#---
# We need to separately store some back-transform statistics for later use
A001_BTS_Float                  = {}
A001_BTS_Float["Name"]          = []
A001_BTS_Float["min_X0"]        = []
A001_BTS_Float["max_X1"]        = []
A001_BTS_Float["LogNormal"]     = []

A001_BTS_nonFloat               = {}
A001_BTS_nonFloat["Name"]       = []
A001_BTS_nonFloat["Type"]       = []
A001_BTS_nonFloat["Quantiles"]  = []

###===###
# Call the helper function
minmax_scaler = MinMaxScaler()

#---
# For every Flt-N2
for itr in range(len(Flt_Variable_N2)):
    
    #---
    # if this is the first variable
    if itr == 0:
    # initialise row number and index number in the DataTypes csv
        Cur_Types_Row = 0
        Cur_Index_Row = 0

    # otherwise
    else:
    # update the row counts
        Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    # Grab the corresponding variable and numpify it
    Cur_Name = Flt_Variable_N2[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))
    # print(list(MyData[Cur_Name]))
    #---
    # then document its properties
    # note, 
    # 	num_classes: 	1
    #   embedding_size: 1
    MyData_Types = MyData_Types.\
                   append({"index":             Cur_Index_Row,
                           "name":              Cur_Name,
                           "type":              "real",
                           "num_classes":       1,
                           "embedding_size":    1,
                           "include":           True,
                           "index_start":       Cur_Types_Row,
                           "index_end":         Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )

    #---
    # Document the back-transformation statistics
    # to be transformed into the range of [0, 1]
    A001_BTS_Float["Name"].append(Cur_Name)

    # re-focus the min value to 0
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())
    # print(Temp_Val)
    # re-scale the max value to 1
    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())
    # print(Temp_Val)
    # Flt-N2 do not need to be logged
    A001_BTS_Float["LogNormal"].append(False)
    # print('so before error， what is going on', Cur_Val)
    #---
    # Save the transformed data in the MyData csv
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    # print(Cur_Val)
    MyData_Transformed[Cur_Name] = Cur_Val
    print(MyData_Transformed[Cur_Name])     # 到这还是正常的
    # tic....tick!
    Cur_Index_Row += 1

#---
# Now iterate through every Flt-LN variables
for itr in range(len(Flt_Variable_LN)):

    #---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    #---
    Cur_Name = Flt_Variable_LN[itr]
    print(Cur_Name)
    Cur_Val  = np.array(list(MyData[Cur_Name]))
    # print(Cur_Val)        怎么这tm有负数啊
    # Logify the variable
    Cur_Val = np.log(Cur_Val + 1)      # 怎么让她不返回0？

    #---
    # Note, 
    # 	num_classes: 	1
    # 	embedding_size: 1
    MyData_Types = MyData_Types.\
                   append({'index'          : Cur_Index_Row,
                           'name'           : Cur_Name,
                           'type'           : 'real',
                           'num_classes'    : 1,
                           'embedding_size' : 1,
                           'include'        : True,
                           'index_start'    : Cur_Types_Row,
                           'index_end'      : Cur_Types_Row + 1
                           },
                          ignore_index = True
                          )
    #---
    A001_BTS_Float["Name"].append(Cur_Name)
        
    Temp_Val = torch.tensor(Cur_Val).view(-1)
    A001_BTS_Float["min_X0"].append(Temp_Val.min().item())

    Temp_Val = Temp_Val - Temp_Val.min()
    A001_BTS_Float["max_X1"].append(Temp_Val.max().item())

    # Flag the variable as logged
    A001_BTS_Float["LogNormal"].append(True)
    # TEST!
    # print(Cur_Val.reshape(-1, 1))
    #---
    Cur_Val = minmax_scaler.fit_transform(Cur_Val.reshape(-1, 1))
    MyData_Transformed[Cur_Name] = Cur_Val
        
    # tic....tic....tick!!
    Cur_Index_Row += 1

#---
# Now iterate through all the Bin variables
for itr in range(len(Bin_Variable)):

    #---
    Cur_Types_Row = list(MyData_Types["index_end"])[-1]

    #---
    Cur_Name = Bin_Variable[itr]
    Cur_Val  = np.array(list(MyData[Cur_Name]))
    # print(Cur_Val)
    #---
    # Note,
    # 	num_classes: 	2
    # 	embedding_size: 2
    # 	index_end: 	Cur_Types_Row + 2
    MyData_Types = MyData_Types.\
                   append({"index":             Cur_Index_Row,
                           "name":              Cur_Name,
                           "type":              "bin",
                           "num_classes":       2,
                           "embedding_size":    2,
                           "include":           True,
                           "index_start":       Cur_Types_Row,
                           "index_end":         Cur_Types_Row + 2
                           },
                          ignore_index = True
                          )

    #---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("bin")

    # Although Bin are non-numeric,
    # no qunatiles needed here
    A001_BTS_nonFloat["Quantiles"].append({})

    #---
    # Transform the non-numeric variables into a machine-readable version
    # For each availabel level (2 in the case for Bin)
    for itr2 in range(2):
        # Creates a column per level, and
        # suffixify the name with _1 or with _2
        Temp_Name = Cur_Name + '_' + str(itr2)

        # If originally of class 1, label 1 in _1, 0 otherwise
        # if originally of class 2, label 1 in _2, 0 otherwise
        Temp_Val  = np.zeros_like(Cur_Val)

    # Find the location of each levels
        Loc_Ele = np.where(Cur_Val == itr2)[0]
    # Oneify the correct locations
        Temp_Val[Loc_Ele] = 1

    # Save the flagged locations of each level in the machine-readable dataset
        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = \
            (MyData_Transformed[Temp_Name]).astype(int)

    # tic....tic....tick**2!**3
    Cur_Index_Row += 1

#---
# Now iterate through all Cat_MTCs
for itr in range(len(Cat_Variable_SIRS)):

    # ---
    Cur_Types_Row = list(MyData_Types['index_end'])[-1]

    # ---
    Cur_Name = Cat_Variable_SIRS[itr]
    Cur_Val = np.floor(np.array(list(MyData[Cur_Name])))

    # ---
    # MTC includes is GCS, SOFA and SIRS
    # GCS: a clinical system designed with minimum 3 and maximum 15 points
    # SOFA: 25 classes
    # SIRS: 5 classes

    MyData_Types = MyData_Types. \
        append({'index': Cur_Index_Row,
                'name': Cat_Variable_SIRS[itr],
                'type': 'cat',
                'num_classes': 5,
                'embedding_size': 4,
                'include': True,
                'index_start': Cur_Types_Row,
                'index_end': Cur_Types_Row + 5
                },
               ignore_index=True
               )
    # ---
    A001_BTS_nonFloat["Name"].append(Cur_Name)
    A001_BTS_nonFloat["Type"].append("SIRS")

    # ---
    # No quantiles needed here
    A001_BTS_nonFloat["Quantiles"].append({})

    # ---
    # You should know what the following loop is for by now...
    for itr2 in range(1, 6):
        Temp_Name = Cur_Name + '_' + str(itr2)
        Temp_Val = np.zeros_like(Cur_Val)

        Loc_Ele = np.where(Cur_Val == itr2)[0]
        Temp_Val[Loc_Ele] = 1

        MyData_Transformed[Temp_Name] = Temp_Val
        MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]). \
            astype(int)

    # ( 0_0)/ toc
    Cur_Index_Row += 1

# for itr in range(len(Cat_Variable_MTC)):
#
#     # ---
#     Cur_Types_Row = list(MyData_Types['index_end'])[-1]
#
#     # ---
#     Cur_Name = Cat_Variable_MTC[itr]
#     Cur_Val = np.floor(np.array(list(MyData[Cur_Name])))
#
#     # ---
#     # MTC includes is GCS, SOFA and SIRS
#     # GCS: a clinical system designed with minimum 3 and maximum 15 points
#     # SOFA: 25 classes
#     # SIRS: 5 classes
#
#     MyData_Types = MyData_Types. \
#         append({'index': Cur_Index_Row,
#                 'name': Cat_Variable_MTC[itr],
#                 'type': 'cat',
#                 'num_classes': 13,
#                 'embedding_size': 4,
#                 'include': True,
#                 'index_start': Cur_Types_Row,
#                 'index_end': Cur_Types_Row + 13
#                 },
#                ignore_index=True
#                )
#     # ---
#     A001_BTS_nonFloat["Name"].append(Cur_Name)
#     A001_BTS_nonFloat["Type"].append("GCS")
#
#     # No quantiles needed here
#     A001_BTS_nonFloat["Quantiles"].append({})
#
#     # ---
#     for itr2 in range(3, 16):
#         Temp_Name = Cur_Name + '_' + str(itr2)
#         Temp_Val = np.zeros_like(Cur_Val)
#
#         Loc_Ele = np.where(Cur_Val == itr2)[0]
#         Temp_Val[Loc_Ele] = 1
#
#         MyData_Transformed[Temp_Name] = Temp_Val
#         MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]). \
#             astype(int)
#
#     # ( 0_0)/ toc
#     Cur_Index_Row += 1

# for itr in range(len(Cat_Variable_SOFA)):
#
#     #---
#     Cur_Types_Row = list(MyData_Types['index_end'])[-1]
#
#     #---
#     Cur_Name  = Cat_Variable_SOFA[itr]
#     Cur_Val  = np.floor(np.array(list(MyData[Cur_Name])))
#
#     #---
#     # MTC includes is GCS, SOFA and SIRS
#     # GCS: a clinical system designed with minimum 3 and maximum 15 points
#     # SOFA: 25 classes
#     # SIRS: 5 classes
#
#     MyData_Types = MyData_Types.\
#                    append({'index'          : Cur_Index_Row,
#                            'name'           : Cat_Variable_SOFA[itr],
#                            'type'           : 'cat',
#                            'num_classes'    : 25,
#                            'embedding_size' : 4,
#                            'include'        : True,
#                            'index_start'    : Cur_Types_Row,
#                            'index_end'      : Cur_Types_Row + 25
#                            },
#                           ignore_index = True
#                           )
#     #---
#     A001_BTS_nonFloat["Name"].append(Cur_Name)
#     A001_BTS_nonFloat["Type"].append("SOFA")

    #---
    # No quantiles needed here
    # A001_BTS_nonFloat["Quantiles"].append({})
    #
    # #---
    # # You should know what the following loop is for by now...
    # for itr2 in range(1, 26):
    #     Temp_Name = Cur_Name + '_' + str(itr2)
    #     Temp_Val  = np.zeros_like(Cur_Val)
    #
    #     Loc_Ele = np.where(Cur_Val == itr2)[0]
    #     Temp_Val[Loc_Ele] = 1
    #
    #     MyData_Transformed[Temp_Name] = Temp_Val
    #     MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
    #                                     astype(int)
    #
    # # ( 0_0)/ toc
    # Cur_Index_Row += 1

#---
# Iterate through the Cat_NLN variables
# We will bin them according to their 10%, 20%, 30% ... etc values
# hence 10 classes
NLN_classes = 10
# #
# for itr in range(len(Cat_Variable_NLN)):
#
#     #---
#     Cur_Types_Row = list(MyData_Types['index_end'])[-1]
#
#     #---
#     Cur_Name = Cat_Variable_NLN[itr]
#     Cur_Val  = np.array(list(MyData[Cur_Name]))
#
#     #---
#     # Note,
#     # 	num_classes: 	NLN_classes
#     # 	embedding_size: 4
#     # 	index_end: 	Cur_Types_Row + NLN_classes
#     MyData_Types = MyData_Types.\
#                    append({'index'          : Cur_Index_Row,
#                            'name'           : Cat_Variable_NLN[itr],
#                            'type'           : 'cat',
#                            'num_classes'    : NLN_classes,  # 10
#                            'embedding_size' : 4,
#                            'include'        : True,
#                            'index_start'    : Cur_Types_Row,
#                            'index_end'      : Cur_Types_Row + NLN_classes
#                            },
#                           ignore_index = True
#                           )
#     #---
#     A001_BTS_nonFloat["Name"].append(Cur_Name)
#     A001_BTS_nonFloat["Type"].append("cat")
#     # the aforementioned 10%, 20%, 30% ... values
#     A001_BTS_nonFloat["Quantiles"].append(
#         [np.quantile(Cur_Val, i/NLN_classes) for i in range(NLN_classes + 1)])
#
#     #---
#     for itr2 in range(NLN_classes):
#         Temp_Name = Cur_Name + '_C' + str(itr2)
#         Temp_Val  = np.zeros_like(Cur_Val)
#
#         Lower_bar = np.quantile(Cur_Val, itr2/NLN_classes)
#         Upper_bar = np.quantile(Cur_Val, (itr2+1)/NLN_classes)
#
#         if itr2 == (NLN_classes - 1):
#             Upper_bar = Upper_bar * 1.05
#
#         Loc_Ele = np.all( [ [Cur_Val >= Lower_bar],
#                             [Cur_Val <  Upper_bar] ],
#                           axis = 0)[0]
#         Temp_Val[Loc_Ele] = 1
#
#         MyData_Transformed[Temp_Name] = Temp_Val
#         MyData_Transformed[Temp_Name] = (MyData_Transformed[Temp_Name]).\
#                                         astype(int)
#     # ( 0_0 )
#     Cur_Index_Row += 1

###===###
# Recalibrate everything one last time for sanity checking
MyData_Types['index']           = (MyData_Types['index']).astype(         int)
MyData_Types['name']            = (MyData_Types['name']).astype(          str)
MyData_Types['type']            = (MyData_Types['type']).astype(          str)
MyData_Types['num_classes']     = (MyData_Types['num_classes']).astype(   int)
MyData_Types['embedding_size']  = (MyData_Types['embedding_size']).astype(int)
MyData_Types['include']         = (MyData_Types['include']).astype(       bool)
MyData_Types['index_start']     = (MyData_Types['index_start']).astype(   int)
MyData_Types['index_end']       = (MyData_Types['index_end']).astype(     int)

# Store the back-transformation statistics
BTS_Folder = "C:/Users/logik/Desktop/rlsepsis234-master - full/data/BTS/"
torch.save(A001_BTS_Float,      BTS_Folder + 'A001_BTS_Float')
torch.save(A001_BTS_nonFloat,   BTS_Folder + 'A001_BTS_nonFloat')

# Store the variable description file
# and the machine-readable transformed ground-truth
Input_Folder = "C:/Users/logik/Desktop/rlsepsis234-master - full/data/"
MyData_Types.to_csv(        Input_Folder + 'A001_DataTypes.csv', index = False)
MyData_Transformed.to_csv(  Input_Folder + 'A002_MyData.csv',    index = False)



















