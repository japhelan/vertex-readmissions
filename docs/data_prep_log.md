# DATA PREP LOG

INITIAL CLEANING STEPS (v0)

1. reformatted all columns and renamed 'readmission_within_30_days' to 'target'
2. Converted nulls in 'excercise_frequency' and 'type_of_treatment' to 'None', as those values were missing in the dataframe but claimed to be present in the provided data dictionary - SOLUTION: issue was read_csv automatically converted 'None' to NA. now fixed ^_^

v1 changes

- removed 80 z-score detected outliers from the age column (all 100+ years by domain knowledge they were determined to be fradulent)
- corrected issue where empty cells were not being filled with Null

v1.2 changes

- removed 10 bmi outliers as well
