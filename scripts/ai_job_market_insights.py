import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.metrics import (classification_report, mean_squared_error, r2_score, accuracy_score,
                             f1_score)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
                              ExtraTreesRegressor, RandomForestClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor

# ---------------------------
# Data Loading and Exploration
# ---------------------------
def load_data(csv_path):
    """
    Loads the CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    print(f"\nDatapoints: {df.shape[0]}, Features: {df.shape[1]}")
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing values in each column:")
    print(df.isna().sum())
    return df

# ---------------------------
# Visualization Functions
# ---------------------------
def plot_remote_friendly(df):
    """
    Plots the count of datapoints for 'Remote_Friendly'.
    """
    df.groupby('Remote_Friendly').size().plot(kind='barh', color=sns.palettes.mpl_palette('Dark2'))
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.title("Remote Friendly Counts")
    plt.tight_layout()
    plt.show()

def plot_company_size_vs_ai_adoption(df):
    """
    Plots a heatmap comparing Company_Size with AI_Adoption_Level.
    """
    # Create a DataFrame where each column represents a Company_Size group
    # and each value is the count of AI_Adoption_Level values.
    df_2dhist = pd.DataFrame({
        x_label: grp['AI_Adoption_Level'].value_counts()
        for x_label, grp in df.groupby('Company_Size')
    })
    plt.figure(figsize=(8, 8))
    sns.heatmap(df_2dhist, cmap='viridis')
    plt.xlabel('Company_Size')
    plt.ylabel('AI_Adoption_Level')
    plt.title("Company Size vs AI Adoption Level")
    plt.tight_layout()
    plt.show()

def plot_job_title_counts(df):
    """
    Plots bar charts for the first 9 categorical features (excluding Salary_USD).
    """
    salary = "Salary_USD"
    # Get list of categorical columns (all except salary)
    cats = [col for col in df.columns if col != salary]
    
    index = 0
    # Plot in groups of 3 columns (3 rows of subplots)
    while index < len(cats):
        n_plots = min(3, len(cats) - index)
        fig, axes = plt.subplots(ncols=n_plots, figsize=(5 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        for i in range(n_plots):
            col = cats[index]
            count = df[col].value_counts()
            count.plot(kind="bar", ax=axes[i])
            # Add labels on top of bars
            for container in axes[i].containers:
                axes[i].bar_label(container)
            axes[i].set_yticklabels([])
            axes[i].set_xlabel("")
            axes[i].set_title(col.replace('_', ' '))
            index += 1
        plt.tight_layout()
        plt.show()

def plot_salary_distribution(df):
    """
    Plots a histogram with KDE for Salary_USD.
    """
    salary = "Salary_USD"
    fig, ax = plt.subplots()
    sns.histplot(df, x=salary, kde=True, ax=ax)
    ax.set_title(salary.replace('_', ' '))
    plt.tight_layout()
    plt.show()

def plot_average_salaries(df):
    """
    Plots average salary for different company types and job titles.
    """
    salary = "Salary_USD"
    cats = [col for col in df.columns if col != salary]
    
    index = 0
    while index < len(cats):
        n_plots = min(3, len(cats) - index)
        fig, axes = plt.subplots(ncols=n_plots, figsize=(5 * n_plots, 6))
        if n_plots == 1:
            axes = [axes]
        for i in range(n_plots):
            col = cats[index]
            grouped = df.groupby(col)
            mean_salary = grouped[salary].mean().sort_values(ascending=False)
            mean_salary.plot(kind="bar", ax=axes[i])
            for container in axes[i].containers:
                axes[i].bar_label(container, rotation=90, label_type="center")
            axes[i].set_yticklabels([])
            axes[i].set_xlabel("")
            axes[i].set_title(col.replace('_', ' '))
            index += 1
        plt.tight_layout()
        plt.show()

def plot_industry_distribution(df):
    """
    Plots the distribution of job roles across different industries.
    """
    plt.figure(figsize=(10, 6))
    order = df['Industry'].value_counts().index
    sns.countplot(y='Industry', data=df, order=order)
    plt.title('Distribution of Job Roles Across Industries')
    plt.tight_layout()
    plt.show()

def plot_remote_work_salary(df):
    """
    Plots a boxplot comparing Remote_Friendly with Salary_USD.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Remote_Friendly', y='Salary_USD', data=df)
    plt.title('Remote Work and Salary')
    plt.tight_layout()
    plt.show()

def plot_salary_by_industry(df):
    """
    Plots a boxplot for salary distribution by Industry.
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Salary_USD', y='Industry', data=df)
    plt.title('Salary Distribution by Industry')
    plt.xlabel('Salary (USD)')
    plt.ylabel('Industry')
    plt.tight_layout()
    plt.show()

def plot_in_demand_skills(df):
    """
    Plots the most in-demand skills based on the 'Required_Skills' column.
    """
    skills = df['Required_Skills'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=skills.values, y=skills.index)
    plt.title('Most In-Demand Skills')
    plt.tight_layout()
    plt.show()

def plot_ai_influence_on_skill_demand(df):
    """
    Plots the influence of AI adoption on required skills.
    """
    skills = df['Required_Skills'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.countplot(y='Required_Skills', hue='AI_Adoption_Level', data=df, order=skills.index)
    plt.title('AI Influence on Skill Demand')
    plt.tight_layout()
    plt.show()

# ---------------------------
# Machine Learning Functions
# ---------------------------
def evaluate_regression_models(df):
    """
    Evaluates multiple regression models to predict Salary_USD.
    The categorical features are label encoded and the data is scaled using MinMaxScaler.
    Model performance is visualized with bar charts for MSE and R2 score.
    """
    salary = "Salary_USD"
    cats = [col for col in df.columns if col != salary]
    
    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in cats:
        df[col] = label_encoder.fit_transform(df[col].values)
    
    # Prepare data for regression
    x = df.drop(salary, axis=1).values
    y = df[salary].values.reshape(-1, 1)
    # Combine features and target then scale
    data = np.hstack((x, y))
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    x = data[:, :-1]
    y = data[:, -1]
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    print(f"Train/Test shapes: {x_train.shape}, {x_test.shape}, {y_train.shape}, {y_test.shape}")
    
    # Define models and their names
    models = [
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        AdaBoostRegressor(),
        ExtraTreesRegressor(),
        SVR(),
        LinearRegression(),
        XGBRegressor(),
        LGBMRegressor(verbose=-100)
    ]
    names = ["Random Forest", "Gradient Boosting", "Ada Boost", "Extra Trees",
             "Support Vector Machine", "Linear Regression", "XGBoost", "LightGBM"]
    
    mses, r2s = [], []
    for model, name in zip(models, names):
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        mses.append(mse)
        r2s.append(r2)
    
    # Create a DataFrame to hold the results
    results_df = pd.DataFrame({"MSE": mses, "R2": r2s}, index=names)
    
    # Plot R2 scores
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    results_df.sort_values("R2", ascending=False)["R2"].plot(kind="bar", ax=axes[0])
    for container in axes[0].containers:
        axes[0].bar_label(container)
    axes[0].set_yticklabels([])
    axes[0].set_title("R2 Score")
    
    # Plot MSE scores
    results_df.sort_values("MSE", ascending=True)["MSE"].plot(kind="bar", ax=axes[1])
    for container in axes[1].containers:
        axes[1].bar_label(container)
    axes[1].set_yticklabels([])
    axes[1].set_title("MSE Score")
    
    plt.tight_layout()
    plt.show()

def automation_risk_prediction(df):
    """
    Performs a simple automation risk prediction.
    Uses one-hot encoding for features (excluding Automation_Risk),
    splits the data, trains a RandomForestClassifier, and prints out accuracy and a classification report.
    """
    X = pd.get_dummies(df.drop(['Automation_Risk'], axis=1), drop_first=True)
    y = df['Automation_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Automation Risk Prediction Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def automation_risk_classification_pipeline(df):
    """
    Builds and evaluates pipelines for predicting Automation_Risk using multiple classifiers.
    The preprocessing step uses a ColumnTransformer for both categorical and numerical features.
    """
    # Reload data in case previous processing changed the DataFrame
    df = pd.read_csv('/content/ai_job_market_insights.csv')
    
    X = df.drop('Automation_Risk', axis=1)
    y = df['Automation_Risk']
    
    categorical_cols = ['Job_Title', 'Industry', 'Company_Size', 'Location',
                          'AI_Adoption_Level', 'Required_Skills',
                          'Remote_Friendly', 'Job_Growth_Projection']
    numeric_cols = ['Salary_USD']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    def evaluate_model(name, model):
        """
        Fits the given model pipeline, predicts, and prints evaluation metrics.
        """
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"=== {name} ===")
        print("Accuracy:", acc)
        print("F1 Score (weighted):", f1)
        print("Classification Report:\n", classification_report(y_test, preds, target_names=label_encoder.classes_))
        print("-" * 40)
        return acc, f1
    
    # Build pipelines for each model
    pipelines = {
        'Logistic Regression': Pipeline([
            ('preprocess', preprocessor),
            ('clf', LogisticRegression(max_iter=500))
        ]),
        'Random Forest': Pipeline([
            ('preprocess', preprocessor),
            ('clf', RandomForestClassifier())
        ]),
        'Support Vector Classifier': Pipeline([
            ('preprocess', preprocessor),
            ('clf', SVC())
        ]),
        'XGBoost': Pipeline([
            ('preprocess', preprocessor),
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        ])
    }
    
    results = {}
    for name, pipeline in pipelines.items():
        acc, f1 = evaluate_model(name, pipeline)
        results[name] = (acc, f1)
    
    # Compare models visually
    model_names = list(results.keys())
    accuracies = [res[0] for res in results.values()]
    f1_scores = [res[1] for res in results.values()]
    
    x_pos = np.arange(len(model_names))
    width = 0.4
    plt.figure(figsize=(8, 5))
    plt.bar(x_pos - width/2, accuracies, width=width, label='Accuracy')
    plt.bar(x_pos + width/2, f1_scores, width=width, label='F1 (weighted)')
    plt.xticks(x_pos, model_names, ha='center')
    plt.ylim([0, 1])
    plt.ylabel('Score')
    plt.title('Model Comparison: Accuracy & F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    csv_path = "data/ai_job_market_insights.csv"
    
    df = load_data(csv_path)
    
    # Visualization Section
    plot_remote_friendly(df)
    plot_company_size_vs_ai_adoption(df)
    plot_job_title_counts(df)
    plot_salary_distribution(df)
    plot_average_salaries(df)
    plot_industry_distribution(df)
    plot_remote_work_salary(df)
    plot_salary_by_industry(df)
    plot_in_demand_skills(df)
    plot_ai_influence_on_skill_demand(df)
    
    # Machine Learning: Regression Models for Salary Prediction
    evaluate_regression_models(df.copy())
    
    # Machine Learning: Automation Risk Prediction (using one-hot encoding)
    automation_risk_prediction(df.copy())
    
    # Machine Learning: Automation Risk Classification Pipeline with multiple models
    automation_risk_classification_pipeline(df.copy())

if __name__ == "__main__":
    main()