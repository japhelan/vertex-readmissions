# DATA PREP LOG

INITIAL CLEANING STEPS

1. reformatted all columns and renamed 'readmission_within_30_days' to 'target'
2. Converted nulls in 'excercise_frequency' and 'type_of_treatment' to 'None', as those values were missing in the dataframe but claimed to be present in the provided data dictionary - SOLUTION: issue was read_csv automatically converted 'None' to NA. now fixed ^_^
