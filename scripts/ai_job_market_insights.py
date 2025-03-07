import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_data(filepath):
    """Loads dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def clean_data(df):
    """Cleans the dataset by handling missing values and duplicates."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['AI_Workload_Ratio'].fillna(df['AI_Workload_Ratio'].mean(), inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Convert 'AI Impact' percentage string to float
    df['AI Impact'] = df['AI Impact'].astype(str).str.rstrip('%').astype(float) / 100
    return df

def visualize_data(df):
    """Creates visualizations including histograms, heatmaps, and pair plots."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
    
    sns.pairplot(df[numeric_cols], diag_kind='kde', height=2.5)
    plt.suptitle("Pairplot of Numerical Variables", y=1.02)
    plt.show()

def train_regression_model(df):
    """Trains a Random Forest model to predict AI Impact."""
    df = df.dropna(subset=['AI Impact'])
    features = ['Tasks', 'AI models', 'AI_Workload_Ratio']
    X = df[features]
    y = df['AI Impact']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R² Score: {r2:.4f}")
    
    # Scatter plot of actual vs. predicted
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual AI Impact")
    plt.ylabel("Predicted AI Impact")
    plt.title("Actual vs. Predicted AI Impact")
    plt.show()
    
    return model

def perform_clustering(df, n_clusters=3):
    """Performs KMeans clustering on the dataset and visualizes results."""
    features = ['Tasks', 'AI models', 'AI_Workload_Ratio']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, df['Cluster'])
    print(f"Silhouette Score: {score:.4f}")
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
    pca_df['Cluster'] = df['Cluster']
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='Set1', s=100, alpha=0.7)
    plt.title("Clusters Visualized with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title='Cluster')
    plt.show()
    
    return df

def main():
    """Main function to run the entire pipeline."""
    df = load_data('data/ai_impact_on_jobs.csv')
    df = clean_data(df)
    visualize_data(df)
    model = train_regression_model(df)
    df = perform_clustering(df)

if __name__ == "__main__":
    main()