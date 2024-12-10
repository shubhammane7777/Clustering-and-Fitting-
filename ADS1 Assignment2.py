import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('/Users/abc/Documents/Applied data science 1 assignment 2/dataset/wine-clustering.csv')

# Check for missing values
print(data.isnull().sum())

# Impute missing values (if necessary)
data.fillna(data.mean(), inplace=True)  # Example: Impute with mean



scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)  # Normalize the data

 # You can use a correlation matrix to see relationships between features


sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()



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



reg = LinearRegression()
reg.fit(X, y)

plt.scatter(X[:, 0], y, label="Data")
plt.plot(X[:, 0], reg.predict(X), color='red', label="Regression Line")
plt.legend()
plt.title("Line Fitting")
plt.show()


def plot_histogram(data, column_name):
    plt.hist(data[column_name], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

# Example usage:
plot_histogram(data, 'Nonflavanoid_Phenols')  

def plot_pie_chart(data, column_name):
    data[column_name].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['red', 'orange', 'yellow'])
    plt.title(f'Pie Chart of {column_name}')
    plt.ylabel('')  # Remove y-label for better aesthetics
    plt.show()

# Example usage:
plot_pie_chart(data[:11], 'Ash_Alcanity')  

