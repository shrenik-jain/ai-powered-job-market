import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load data from a csv file

    Args:
        csv_path (str): the path to the csv file

    Returns:
        df (pd.DataFrame): a pandas dataframe
    """
    # validate the input
    assert isinstance(csv_path, str), "csv_path must be a string"
    assert csv_path.endswith(".csv"), "csv_path must be a path to a csv file"

    df = pd.read_csv(csv_path)
    print("\nData shape: ", df.shape)
    print("\nData columns: ", df.columns)
    print("\n Statistics of the Data")
    print(df.describe())
    return df


def preprocess_data(df: pd.DataFrame) -> tuple:
    """
    Preprocess the data by encoding the categorical variables, scaling the data and 
    splitting the data into training and testing sets

    Args:
        df (pd.DataFrame): the dataframe to be preprocessed
    
    Returns:
        x_train, x_test, y_train, y_test (np.ndarray): the training and testing sets
    """
    # validate the input
    assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"
    assert df.shape[0] > 0, "df must not be empty"
    assert df.shape[1] > 1, "df must have more than one column"

    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    salary = "Salary_USD"
    cats = [i for i in df.columns if i != salary]
    for i in cats:
        df[i] = label_encoder.fit_transform(df[i].values)

    x = df.drop(salary, axis=1).values
    y = df[salary].values
    y = y.reshape(-1, 1)

    data = np.hstack((x, y))
    data = scaler.fit_transform(data)

    x = data[:, :-1]
    y = data[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    return (x_train, x_test, y_train, y_test)


def load_models() -> tuple:
    """
    Load the models to be used for training

    Returns:
        models (list): a list of models
        names (list): a list of the names of the models
    """
    rfc = RandomForestRegressor()
    gbc = GradientBoostingRegressor()
    abc = AdaBoostRegressor()
    etc = ExtraTreesRegressor()
    svr = SVR()
    lnr = LinearRegression()
    xgb = XGBRegressor()
    lgb = LGBMRegressor(verbose=-100)

    models = [rfc, gbc, abc, etc, svr, lnr, xgb, lgb]
    names = ["Random Forest", "Gradient Boosting", "Ada Boost", "Extra Trees",
            "Support Vector Machine", "Linear Regression", "XGBoost", "LightGBM"]
    
    return (models, names)


def training(models: list, names: list, x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Train the models and evaluate their performance using the mean squared error and r2 score

    Args:
        models (list): a list of models
        names (list): a list of the names of the models
        x_train (np.ndarray): the training data
        x_test (np.ndarray): the testing data
        y_train (np.ndarray): the training labels
        y_test (np.ndarray): the testing labels
    """
    # validate the input
    assert isinstance(models, list), "models must be a list"
    assert isinstance(names, list), "names must be a list"
    assert len(models) == len(names), "models and names must have the same length"
    assert isinstance(x_train, np.ndarray), "x_train must be a numpy array"
    assert isinstance(x_test, np.ndarray), "x_test must be a numpy array"
    assert isinstance(y_train, np.ndarray), "y_train must be a numpy array"
    assert isinstance(y_test, np.ndarray), "y_test must be a numpy array"

    mses, r2s = [], []
    for i, j in zip(models, names):
        i.fit(x_train, y_train)
        pred = i.predict(x_test)
        mses += [mean_squared_error(pred, y_test)]
        r2s += [r2_score(pred, y_test)]

    dd = pd.DataFrame({"mse": mses, "r2": r2s}, index=names)
    fig, axes = plt.subplots(ncols=2, figsize=(15, 6))
    index = 0

    dd = dd.sort_values("r2", ascending=False)
    dd["r2"].plot(kind="bar", ax=axes[index])
    for container in axes[index].containers:
        axes[index].bar_label(container)
    axes[index].set_yticklabels(())
    axes[index].set_xlabel("")
    axes[index].set_ylabel("")
    axes[index].set_title("R2 score")
    index += 1
    dd = dd.sort_values("mse", ascending=True)
    dd["mse"].plot(kind="bar", ax=axes[index])

    for container in axes[index].containers:
        axes[index].bar_label(container)
    axes[index].set_yticklabels(())
    axes[index].set_xlabel("")
    axes[index].set_ylabel("")
    axes[index].set_title("MSE score")

    plt.tight_layout()
    plt.show()


def train_classifier(df: pd.DataFrame) -> tuple:
    """
    Train a classifier to predict the Automation Risk

    Args:
        df (pd.DataFrame): the dataframe to be used for training
    """
    assert isinstance(df, pd.DataFrame), "df must be a pandas dataframe"
    assert df.shape[0] > 0, "df must not be empty"
    assert df.shape[1] > 1, "df must have more than one column"

    X = pd.get_dummies(df.drop(['Automation_Risk'], axis=1), drop_first=True)
    y = df['Automation_Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return (model, acc, report)

if __name__ == "__main__":
    # Load the data
    csv_path = "data/ai_job_market_insights.csv"
    df = load_data(csv_path)
    print("Data Loaded Successfully")
    print("\nData head: ", df.head())
    print("="*50)
    # Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data(df)
    print("\nData Preprocessed Successfully")
    print("\nTraining and Testing Data Shapes")
    print("x_train: ", x_train.shape)
    print("x_test: ", x_test.shape)
    print("y_train: ", y_train.shape)
    print("y_test: ", y_test.shape)
    print("="*50)
    print("\nTraining the Regression Models")
    # Train the regression models
    training(*load_models(), x_train, x_test, y_train, y_test)
    print("="*50)
    print("\nTraining the Classifier")
    # Train the classifier model
    model, acc, report = train_classifier(df)
    print("\nAccuracy: ", acc)
    print("\nClassification Report: ", report)
    print("="*50)

