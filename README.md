# Crypto
# Overview
This Jupyter Notebook provides a starter code for performing cluster analysis on a cryptocurrency dataset using Principal Component Analysis (PCA) and K-Means clustering. The analysis helps in visualizing and interpreting clusters based on various features, reducing the dimensionality of the data, and optimizing clustering performance.

# Contents
The notebook includes the following key sections:

1. Import Libraries: Imports essential Python libraries for data manipulation, visualization, and machine learning, such as pandas, numpy, matplotlib, and sklearn.
2. Load Data: Loads the cryptocurrency data from a CSV file and displays the first few rows to get an overview of the dataset.
3. Preprocess Data: Scales the features using standard scaling methods to normalize the data, making it suitable for clustering.
4. PCA Implementation: Performs Principal Component Analysis to reduce the dataset to a specified number of principal components.
5. Clustering with K-Means: Applies the K-Means clustering algorithm on the original and PCA-reduced datasets to identify clusters.
6. Visualize Results: Creates composite plots to visualize and contrast the clustering results in both the original feature space and the PCA-reduced space.
7. Analyze Explained Variance: Calculates and interprets the explained variance for the principal components used.

# Instructions
Install Required Packages:
1. Ensure we have the required packages installed. We install missing packages using pip:
pip install pandas numpy matplotlib scikit-learn
2. Load the Notebook:
Open the Jupyter Notebook using JupyterLab or Jupyter Notebook:
jupyter notebook Crypto_Clustering_starter_code.ipynb
3. Run Cells Sequentially:
Execute the notebook cells sequentially. Each cell builds upon the previous one.
4. Key Functions and Methods
PCA: Used for dimensionality reduction by identifying principal components that capture the most variance in the data.
KMeans: Clustering algorithm that partitions the data into clusters by minimizing the variance within each cluster.
fit_transform: Fits the model to the data and transforms it according to the fitted model.
explained_variance_ratio_: Attribute that provides the amount of variance explained by each principal component.
The notebook generates:
5. Clustered Dataframes: Dataframes with cluster labels for both original and reduced datasets.
Visual Plots: Composite scatter plots that visually contrast the clusters in different feature spaces.

# Output
 After visually analyzing the cluster analysis results, using fewer features for clustering, as shown in the plots, often has significant impacts on the clustering results. Here's an analysis based on the visualizations:

Feature Reduction Impact:

Original Features (Left Plot): The left plot visualizes clustering based on the original features (price_change_percentage_24h and price_change_percentage_7d). Here, data points are more spread out, and clusters are more distinct in the feature space. 

Reduced Features (Right Plot): The right plot uses principal components (PCA1 and PCA2) obtained through Principal Component Analysis (PCA). Clustering on these fewer features (reduced dimensions) shows a more compact view of clusters. The clusters are aligned along the principal components, which capture the maximum variance in the data.
Advantages of Fewer Features:

1. Simplification: Using fewer features simplifies the model, reducing dimensionality while preserving most of the variance in the data. This often makes it easier to visualize and interpret clusters.
2. Performance: With fewer features, the clustering algorithm (like K-Means) can operate more efficiently, requiring less computational power and time.
3. Noise Reduction: Reducing features can help eliminate noise and irrelevant information, leading to more robust and distinct clusters.
Potential Drawbacks:

1. Loss of Information: Some specific details and variations present in the original feature space might be lost during dimensionality reduction, potentially making clusters less specific.
2. Interpretability: Clusters in reduced dimensions might be harder to interpret in terms of original features, as principal components are linear combinations of original features and do not have direct physical meaning.

Conclusion:
Using fewer features through PCA results in a more compact representation of clusters, preserving most of the important variance and potentially simplifying the clustering task. However, this comes at the cost of losing some detail from the original feature space. 
