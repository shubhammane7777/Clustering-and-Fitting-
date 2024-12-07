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

from sklearn.cluster import KMeans

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

optimal_k = 3  # Assume 3 clusters from Elbow Method or silhouette score
kmeans = KMeans(n_clusters=optimal_k)
clusters = kmeans.fit_predict(data_scaled)

plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis')
plt.title("K-Means Clustering")
plt.show()

X = data_scaled[:, :-1]  # Example: All columns except the last one
y = data_scaled[:, -1]   # Example: The last column as the target variable

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X, y)

plt.scatter(X[:, 0], y, label="Data")
plt.plot(X[:, 0], reg.predict(X), color='red', label="Regression Line")
plt.legend()
plt.title("Line Fitting")
plt.show()


