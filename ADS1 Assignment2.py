# Import necessary libraries
import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = '/Users/abc/Documents/Applied data science 1 assignment 2/dataset/wine-clustering.csv'  # Replace with your dataset's file path
data = pd.read_csv(file_path)

# Step 1: Handle Missing Values
# Check for missing values
print("Missing Values Per Column:\n", data.isnull().sum())

# Option to remove rows/columns with missing values (if applicable)
# data = data.dropna()  # Drop rows with missing values
# data = data.dropna(axis=1)  # Drop columns with missing values

# Option to impute missing values (if applicable)
# data.fillna(data.mean(), inplace=True)

# Step 2: Normalize/Scale Data
# Initialize the scaler
scaler = StandardScaler()

# Scale the data
scaled_data = scaler.fit_transform(data)

# Convert the scaled data back to a DataFrame for easy readability
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

# Display scaled data statistics to confirm scaling
print("Scaled Data Summary:\n", scaled_df.describe())

# Step 3: Feature Selection
# Retain relevant features for clustering and regression (modify as needed)
# Here we assume all features are relevant, but you can adjust this based on domain knowledge or correlation analysis
selected_features = scaled_df  # Change this if specific features need selection

# Output the processed data
print("Processed Data Sample:\n", selected_features.head())

# Save processed data if needed
selected_features.to_csv('processed_dataset.csv', index=False)
