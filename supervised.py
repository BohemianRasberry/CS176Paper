from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

# LOADING DATASET
df = pd.read_csv("Mesothelioma-data.csv")

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
# print(df.isnull().sum())

# Remove irrelevant variables
df_semi_clean = df.drop(['keep side', 'city', 'asbestos exposure'], axis=1)

df_semi_clean.rename(columns={
    'age': 'age',
    'gender': 'gender',
    'type of MM': 'mesotheliomaType',
    'duration of asbestos exposure': 'asbestosExposure',
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


'''
# Checking on outliers
z_scores = np.abs(stats.zscore(df_semi_clean))
outlier_indices = np.where(z_scores > 3)  # Assuming 3 as the Z-score threshold
unique_row_indices = np.unique(outlier_indices[0])

print("Rows with outliers:", unique_row_indices)

# Removing outliers and cleaning data set
df_clean = df_semi_clean.drop(unique_row_indices)
df_clean.reset_index(drop=True, inplace=True)

'''

# Convert Floats to Integers
for column in df_semi_clean.select_dtypes(include=['float']):
    # Convert only if the float values are essentially integers
    if all(df_semi_clean[column] % 1 == 0):
        df_semi_clean[column] = df_semi_clean[column].astype(int)

df_clean = df_semi_clean

df_clean.to_csv('Mesothelioma_clean_data.csv', index = False)

print(df_clean.describe())
print(df_clean.dtypes)

'''
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

'''

# SPLIT DATASET
X = df_clean.drop('diagnosisClass', axis=1)
y = df_clean['diagnosisClass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DATA TRANSFORMATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# FEATURE SELECTION
rfr = RandomForestRegressor()
rfe = RFE(estimator=rfr, n_features_to_select=5)
rfe.fit(X_train_scaled, y_train)

# print("Selected features: ", rfe.support_)
# Identifying which features were selected
selected_features = X.columns[rfe.support_]
print("Selected features: ", selected_features)
print("Feature ranking: ", rfe.ranking_)

# Refit SVR on selected features in the training data
X_train_rfe = rfe.transform(X_train_scaled)
X_test_rfe = rfe.transform(X_test_scaled)

# Model Training with SVR
svr = SVR(kernel='rbf')  # 'rbf' is used for non-linear problems, change if needed
svr.fit(X_train_rfe, y_train)

# MODEL EVALUATION
y_pred = svr.predict(X_test_rfe)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)