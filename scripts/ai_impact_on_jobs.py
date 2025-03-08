import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_data(filepath):
    """
    Load the dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())
    print(df.describe())
    return df

def data_cleaning(df):
    """
    Cleans the data by handling infinite values and missing values
    in the 'AI_Workload_Ratio' column.
    """
    df['AI_Workload_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Null Values before handling:\n", df.isnull().sum())
    
    mean_val = df['AI_Workload_Ratio'].mean()
    df['AI_Workload_Ratio'].fillna(mean_val, inplace=True)
    
    print("Null Values after handling:\n", df.isnull().sum())
    print('*' * 30)
    print("Number of duplicates:", df.duplicated().sum())
    
def transform_ai_impact(df):
    """
    Transforms the 'AI Impact' column from a percentage string to a float.
    """
    df['AI Impact'] = df['AI Impact'].astype(str).str.rstrip('%').astype(float) / 100

def plot_stripplot(df):
    """
    Creates a strip plot for AI Impact vs Domain.
    """
    sns.stripplot(x="AI Impact", y="Domain", data=df)
    plt.title("Stripplot of AI Impact vs Domain")
    plt.tight_layout()
    plt.show()

def plot_histograms(df):
    """
    Plots histograms with KDE for each numerical column.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

def plot_heatmap(df):
    """
    Plots a heatmap for the correlation of numerical features.
    """
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_pairplot(df):
    """
    Creates a pairplot for all numerical columns.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    sns.pairplot(df[numeric_cols], diag_kind='kde', height=2.5)
    plt.suptitle("Pairplot of Numerical Variables", y=1.02)
    plt.tight_layout()
    plt.show()

def regression_task(df):
    """
    Prepares the data and runs a regression task to predict AI Impact.
    It performs encoding for 'Job titiles' and 'Domain', scales features,
    splits the data, trains a RandomForest model, and evaluates its performance.
    """
    # Create encoded features based on group means
    df["Job_Title_Encoded"] = df.groupby("Job titiles")["AI Impact"].transform("mean")
    df["Domain_Encoded"] = df.groupby("Domain")["AI Impact"].transform("mean")
    
    # Select relevant features
    processed_df = df[['Job_Title_Encoded', 'Domain_Encoded', 'Tasks', 'AI models', 'AI_Workload_Ratio', 'AI Impact']]
    
    # Handle infinity values and missing values
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_df.fillna(processed_df.median(), inplace=True)
    
    # Scale numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    features_to_scale = ['Tasks', 'AI models', 'AI_Workload_Ratio', 'Job_Title_Encoded', 'Domain_Encoded']
    processed_df[features_to_scale] = scaler.fit_transform(processed_df[features_to_scale])
    
    # Split data into features and target
    X = processed_df.drop(columns=['AI Impact'])
    y = processed_df['AI Impact']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest Regression Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nRegression Model Performance:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R² Score: {r2}")
    
    # Plot Actual vs Predicted AI Impact
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.xlabel("Actual AI Impact")
    plt.ylabel("Predicted AI Impact")
    plt.title("Actual vs Predicted AI Impact")
    plt.tight_layout()
    plt.show()

def clustering_task(df):
    """
    Executes a clustering task:
    - Scales selected features ('Tasks', 'AI models', 'AI_Workload_Ratio')
    - Applies KMeans clustering
    - Calculates the silhouette score
    - Uses PCA to reduce dimensions for visualization
    - Plots clusters and generates various visualizations.
    """
    features = ['Tasks', 'AI models', 'AI_Workload_Ratio']
    X = df[features]
    
    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering (using 3 clusters as an example)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = cluster_labels
    
    # Evaluate clustering performance using the silhouette score
    score = silhouette_score(X_scaled, cluster_labels)
    print("Silhouette Score:", score)
    
    # Reduce dimensions with PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = cluster_labels
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set1', s=100, alpha=0.7)
    plt.title("Clusters Visualized with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()
    
    # Compute and print mean values of each cluster for interpretation
    cluster_summary = df.groupby('Cluster')[features].mean()
    print("Cluster Mean Values:\n", cluster_summary)
    
    # Boxplots for feature distributions by cluster
    plt.figure(figsize=(12, 5))
    for i, feature in enumerate(features):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(x='Cluster', y=feature, data=df, palette='Set2')
        plt.title(f"Distribution of {feature} by Cluster")
    plt.tight_layout()
    plt.show()
    
    # Density plots for each feature by cluster
    plt.figure(figsize=(12, 5))
    for i, feature in enumerate(features):
        plt.subplot(1, 3, i + 1)
        for cluster in df['Cluster'].unique():
            sns.kdeplot(df[df['Cluster'] == cluster][feature], label=f'Cluster {cluster}', fill=True)
        plt.title(f"Density Plot of {feature}")
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Pairplot to visualize feature relationships within clusters
    sns.pairplot(df, hue="Cluster", vars=features, palette='Set1')
    plt.suptitle("Pairplot of Features Colored by Cluster", y=1.02)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to load data, clean and transform it, and perform visualizations
    and machine learning tasks on the AI Impact on Jobs dataset.
    """
    filepath = 'data/ai_impact_on_jobs.csv'
    df = load_data(filepath)
    
    # Data cleaning and transformation
    data_cleaning(df)
    transform_ai_impact(df)
    
    # Visualization
    plot_stripplot(df)
    plot_histograms(df)
    plot_heatmap(df)
    plot_pairplot(df)
    
    # Regression task
    regression_task(df.copy())
    
    # Clustering task
    clustering_task(df.copy())

if __name__ == "__main__":
    main()