import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/abc/Documents/Applied data science 1 assignment 2/dataset/wine-clustering.csv')

# Check for missing values
print(data.isnull().sum())

# Impute missing values (if necessary)
data.fillna(data.mean(), inplace=True)  # Example: Impute with mean

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Normalize the data

 # You can use a correlation matrix to see relationships between features
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
