from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from scipy import stats
import numpy as np
import pandas as pd

# LOADING DATASET
df = pd.read_csv("Mesothelioma-data.csv")

# Check for missing values
print(df.isnull().sum())

# Remove irrelevant variables
df_semi_clean = df.drop(['keep side'], axis=1)

df_semi_clean.rename(columns={
    'age': 'age',
    'gender': 'gender',
    'city': 'city',
    'asbestos exposure': 'asbestosExposure',
    'type of MM': 'mesotheliomaType',
    'duration of asbestos exposure': 'asbestosExposureDuration',
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

# Removing outliers and cleaning data set
df_clean = df_semi_clean.drop(unique_row_indices)
df_clean.reset_index(drop=True, inplace=True)

print(df_clean.head())
print(df_clean.info())

df_clean.to_csv('Mesothelioma_clean_data.csv', index = False)

# print(df_clean.describe())
# print(df_clean.dtypes)

# Logistic Regression Model overfits the data

X = df_clean.drop('diagnosisClass', axis=1)
y = df_clean['diagnosisClass']

# Scaling the data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training to Test Data Split 80 to 20
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

'''
# Using SMOTE to balance the data set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Using SMOTE and TomekLinks to balance the data
smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)
'''

# Initialize the Logistic Regression model
log_reg = LogisticRegression(solver='liblinear', random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)
score_ = log_reg.score(X_test, y_test)

# Evaluate the model
print('RESULTS FOR LOGREG')
print('Classification Report: ')
print(classification_report(y_test, y_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred))
print('\nAccuracy Score: ', accuracy_score(y_test, y_pred))