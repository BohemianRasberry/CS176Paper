# Column Name Changes (due to mispellings) #
column_rename = {'age':'Age', 'gender':'Gender', 'city':'City', 'asbestos exposure':'Asbestos Exposure', 'type of MM':'MM type', 
              'duration of asbestos exposure':'Asbestos Exposure (Duration)', 'diagnosis method':'Diagnosis Method', 'keep side':'Keep Side',
              'cytology':'Cytology', 'duration of symptoms':'Symptoms (Duration)', 'dyspnoea':'Dyspnea', 'ache on chest':'Chest Ache', 
              'weakness':'Weakness', 'habit of cigarette':'Cigarette Habit', 'performance status':'Performance Status', 
              'white blood':'White Blood Cells', 'cell count (WBC)':'Cell Count (WBC)', 'hemoglobin (HGB)':'Hemoglobin (HGB)', 
              'platelet count (PLT)':'Platelet Count (PLT)', 'sedimentation':'Sedimentation', 'blood lactic dehydrogenise (LDH)':'Blood Lactic Dehydrogenise (LDH)',
              'alkaline phosphatise (ALP)':'Alkaline Phosphatise (ALP)','total protein':'Protein Total', 'albumin':'Albumin', 'glucose':'Glucose', 
              'pleural lactic dehydrogenise':'Pleural Lactic Dehydrogenase', 'pleural protein':'Pleural Protein', 'pleural albumin':'Pleural Albumin', 
              'pleural glucose':'Pleural glucose', 'dead or not':'Status', 'pleural effusion':'Pleural Effusion', 
              'pleural thickness on tomography':'Pleural Thickness (Tomography)', 'pleural level of acidity (pH)':'Pleural Level of Acidity (pH)', 
              'C-reactive protein (CRP)':'C-reactive Protein (CRP)', 'class of diagnosis':'Class of Diagnosis'}

# Pandas #
import pandas as pd

# Read and DataFrame Creation #
df = pd.read_csv('Mesothelioma-data.csv')
df.rename(columns=column_rename, inplace=True)
mesothelioma_dataset = pd.DataFrame(df)
mesothelioma_dataset['Class of Diagnosis'].replace(1, 'Healthy',inplace=True)
mesothelioma_dataset['Class of Diagnosis'].replace(2, 'Mesothelioma',inplace=True)

# Print #
print("")
print(mesothelioma_dataset.head())
print("")

from sklearn.preprocessing import StandardScaler

mesothelioma_dataset['Class of Diagnosis'].replace('Healthy', 1, inplace=True)
mesothelioma_dataset['Class of Diagnosis'].replace('Mesothelioma', 2, inplace=True)
new_column_names = list(column_rename.values())
x = mesothelioma_dataset[new_column_names].values
x = StandardScaler().fit_transform(x)

import numpy as np

print("Mean & Standard Deviation:" "(", np.mean(x) , "," , np.std(x) , ")")
print("")

feature_columns = ['feature'+str(i) for i in range(x.shape[1])]
normalized_mesothelioma = pd.DataFrame(x,columns=feature_columns)

# Print #
print(normalized_mesothelioma.head())
print("")

from sklearn.decomposition import PCA

pca_mesothelioma = PCA(n_components=2)
principalComponents_mesothelioma = pca_mesothelioma.fit_transform(x)
principal_mesothelioma_Df = pd.DataFrame(data = principalComponents_mesothelioma
             , columns = ['principal component 1', 'principal component 2'])
explained_variances = pca_mesothelioma.explained_variance_ratio_

# Print #
print(principal_mesothelioma_Df.head())
print("")
print('Explained variation per principal component: {}'.format(explained_variances))

import matplotlib.pyplot as plt

variance_pc1 = explained_variances[0]
variance_pc2 = explained_variances[1]

df['Class of Diagnosis'].replace(1, 'Healthy',inplace=True)
df['Class of Diagnosis'].replace(2, 'Mesothelioma',inplace=True)

plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel(f'Principal Component - 1: {variance_pc1:.2f}',fontsize=20)
plt.ylabel(f'Principal Component - 2: {variance_pc2:.2f}',fontsize=20)
plt.title("Principal Component Analysis of Mesotheliama Cancer Dataset",fontsize=20)
targets = ['Healthy', 'Mesothelioma']
colors = ['g', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = df['Class of Diagnosis'] == target
    plt.scatter(principal_mesothelioma_Df.loc[indicesToKeep, 'principal component 1']
               , principal_mesothelioma_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.show()