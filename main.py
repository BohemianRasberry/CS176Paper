import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# DATA CLEANING
df = pd.read_csv("Mesothelioma-data.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
# print(df.isnull().sum())

# Remove irrelevant variables
df_semi_clean = df.drop(['keep side', 'city'], axis=1)

df_semi_clean.rename(columns={
    'age': 'age',
    'gender': 'gender',
    'asbestos exposure': 'asbestosExposure',
    'type of MM': 'mesotheliomaType',
    'duration of asbestos exposure': 'durationOfAsbestosExposure',
    'diagnosis method': 'diagnosisMethod',
    'cytology': 'cytology',
    'duration of symptoms': 'durationOfSymptoms',
    'dyspnoea': 'dyspnoea',
    'ache on chest': 'chestAche',
    'weakness': 'weakness',
    'habit of cigarette': 'cigaretteHabit',
    'performance status': 'performaceStatus',
    'white blood': 'whiteBlood',
    'cell count (WBC)': 'wbcCount',
    'hemoglobin (HGB)': 'hbg',
    'platelet count (PLT)': 'plt',
    'sedimentation': 'esr',
    'blood lactic dehydrogenise (LDH)': 'ldh',
    'alkaline phosphatise (ALP)': 'alp',
    'total protein': 'totalProtein',
    'albumin': 'albumin',
    'glucose': 'glucose',
    'pleural lactic dehydrogenise': 'pld',
    'pleural protein': 'pleuralProtein',
    'pleural albumin': 'pleuralAlbumin',
    'pleural glucose': 'pleuralGlucose',
    'dead or not': 'mortality',
    'pleural effusion': 'pleuralEffusion',
    'pleural thickness on tomography': 'pleuralThickness',
    'pleural level of acidity (pH)': 'pleuralAcidity',
    'C-reactive protein (CRP)': 'crp',
    'class of diagnosis': 'diagnosisClass'
}, inplace=True)

print(df_semi_clean.head())

# Removed duplicates
df_semi_clean = df_semi_clean.drop_duplicates()

# Checking on outliers
z_scores = np.abs(stats.zscore(df_semi_clean))
outlier_indices = np.where(z_scores > 3)  # Assuming 3 as the Z-score threshold
unique_row_indices = np.unique(outlier_indices[0])

print("Rows with outliers:", unique_row_indices)

# Convert Floats to Integers
for column in df_semi_clean.select_dtypes(include=['float']):
    # Convert only if the float values are essentially integers
    if all(df_semi_clean[column] % 1 == 0):
        df_semi_clean[column] = df_semi_clean[column].astype(int)

# Data Type Conversion
df_semi_clean['gender'] = df_semi_clean['gender'].astype('category')
df_semi_clean['mesotheliomaType'] = df_semi_clean['mesotheliomaType'].astype('category')
# df_semi_clean['mesotheliomaType'] = df_semi_clean['mesotheliomaType'].astype('category')

# Removing outliers and cleaning data set
df_clean = df_semi_clean.drop(unique_row_indices)
df_clean.reset_index(drop=True, inplace=True)

df_clean.to_csv('Mesothelioma_clean_data.csv', index = False)

print(df_clean.describe())
print(df_clean.dtypes)

# DATA TRANSFORMATION

scaler = StandardScaler()

df_clean[['asbestosExposure',
          'durationOfSymptoms',
          'whiteBlood',
          'wbcCount',
          'plt',
          'esr',
          'ldh',
          'alp',
          'totalProtein',
          'albumin',
          'glucose',
          'pld',
          'pleuralProtein',
          'pleuralAlbumin',
          'pleuralGlucose',
          'crp']] = scaler.fit_transform(df_clean[['asbestosExposure',
          'durationOfSymptoms',
          'whiteBlood',
          'wbcCount',
          'plt',
          'esr',
          'ldh',
          'alp',
          'totalProtein',
          'albumin',
          'glucose',
          'pld',
          'pleuralProtein',
          'pleuralAlbumin',
          'pleuralGlucose',
          'crp'
          ]])

df_clean.to_csv('Mesothelioma_transform_data.csv', index = False)

## VISUALIZATION OF UNSTRANSFORMED TRANSFORMED DATA

columns_to_visualize = ['whiteBlood', 'wbcCount', 'plt']

# UNSTRANSFORMED DATA
plt.figure(figsize=(15, 5))
for i, column in enumerate(columns_to_visualize, 1):
    plt.subplot(1, len(columns_to_visualize), i)
    sns.boxplot(y=df_semi_clean[column])
    plt.title(f'Before: {column}')

# Box plots after transformation
plt.figure(figsize=(15, 5))
for i, column in enumerate(columns_to_visualize, 1):
    plt.subplot(1, len(columns_to_visualize), i)
    sns.boxplot(y=df_clean[column], color='orange')
    plt.title(f'After: {column}')

plt.show()

# DATA