import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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

 # we can use a correlation matrix to see relationships between features


def plot_correlation_matrix(data, title="Correlation Matrix", cmap='coolwarm', annot=True):
    """
    Plots a heatmap to display the correlation matrix of a given dataset.
    
    Parameters:
    - data (pd.DataFrame): The dataset for which the correlation matrix is calculated.
    - title (str): The title of the heatmap. Default is "Correlation Matrix".
    - cmap (str): The colormap used for the heatmap. Default is 'coolwarm'.
    - annot (bool): Whether to annotate the heatmap with correlation coefficients. Default is True.
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt=".2f")
    plt.title(title, fontsize=14)
    plt.show()

 plot_correlation_matrix(data)

def plot_elbow_method(data, max_clusters=10, title="Elbow Method for Optimal Clusters"):
    """
    Plots the Elbow Method to determine the optimal number of clusters.
    
    Parameters:
    - data (np.ndarray or pd.DataFrame): The scaled data to cluster.
    - max_clusters (int): The maximum number of clusters to test. Default is 10.
    - title (str): The title of the plot. Default is "Elbow Method for Optimal Clusters".
    """
    wcss = []  # Within-cluster sum of squares
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducibility
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-', color='b')
    plt.title(title, fontsize=14)
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    plt.show()

# Example usage:
plot_elbow_method(data_scaled)

def plot_histogram(data, column_name):
    plt.hist(data[column_name], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

plot_histogram(data, 'Nonflavanoid_Phenols')  

# Choose features for regression (example: Feature_1 and Feature_2 from dataset)
X = data_scaled[:, 0].reshape(-1, 1)  # Independent variable
y = data_scaled[:, 1]  # Dependent variable

# Fit and visualize regression line
def plot_linear_regression(X, y, title="Line Fitting"):
    """
    Fits a Linear Regression model to the data and visualizes the regression line.
    
    Parameters:
    - X (np.ndarray or pd.DataFrame): The independent variable(s).
    - y (np.ndarray or pd.Series): The dependent variable.
    - title (str): The title of the plot. Default is "Line Fitting".
    """
    reg = LinearRegression()
    reg.fit(X, y)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, label="Data", color="blue", alpha=0.6)
    plt.plot(X, reg.predict(X), color='red', label="Regression Line")
    plt.title(title, fontsize=14)
    plt.xlabel("Flavanoids", fontsize=12)
    plt.ylabel("Nonflavanoid_Phenols", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.show()

plot_linear_regression(X, y)

silhouette_avg = silhouette_score(data_scaled, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

def plot_kmeans_clustering(data, optimal_k, title="K-Means Clustering"):
    """
    Applies K-Means clustering to the data and visualizes the clusters.
    
    Parameters:
    - data (np.ndarray or pd.DataFrame): The scaled data for clustering.
    - optimal_k (int): The number of clusters to use for K-Means.
    - title (str): The title of the scatter plot. Default is "K-Means Clustering".
    """
    # Fit K-Means model
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)  # Set random_state for reproducibility
    clusters = kmeans.fit_predict(data)
    
    # Scatter plot of the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', edgecolor='k', s=50)
    plt.title(title, fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.grid(True)
    plt.colorbar(label='Cluster')
    plt.show()

# Example usage:
plot_kmeans_clustering(data_scaled, optimal_k=3)
    
