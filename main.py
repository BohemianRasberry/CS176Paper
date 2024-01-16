import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mesothelioma-data.csv")

# Display the first few rows of the dataset
# print(df.head())

# Get a concise summary of the DataFrame
# print(df.info())

# Descriptive statistics for numerical columns
# print(df.describe())

# Check for missing values
# print(df.isnull().sum())



# Example: Identifying outliers using Z-score
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))

print(z_scores)

outliers = np.where(z_scores > 3)

# Depending on the context, you might want to inspect these outliers before deciding to remove them.
print(outliers)