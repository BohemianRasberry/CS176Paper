import pandas as pd
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

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

df = pd.read_csv('Mesothelioma-data.csv')

df.rename(columns=column_rename, inplace=True)

X = df.data
y = df.target

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()


