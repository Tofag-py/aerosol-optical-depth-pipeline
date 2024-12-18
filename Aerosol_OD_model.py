import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import plotly.express as px
from category_encoders import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prefect import task, flow
from ingestion import main_flow
from prefect.logging import get_run_logger


@task()
def logger_task():
    # this logger instance will emit logs 
    # associated with both the flow run *and* the individual task run
    logger = get_run_logger()
    logger.info("INFO level log message from a task.")
    logger.debug("DEBUG level log message from a task.")


@flow()
def logger_flow():
    # this logger instance will emit logs
    # associated with the flow run only
    logger = get_run_logger()
    logger.info("INFO level log message.")
    logger.debug("DEBUG level log message.")
    logger.error("ERROR level log message.")
    logger.critical("CRITICAL level log message.")

    logger_task()


@task
def create_dataframe(path:str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df



def wrangle(df: pd.DataFrame):
    """
    Cleans and prepares the input DataFrame.
    - Removes outliers from 'Aerosol_OD_Mean' based on the 10th and 90th percentiles.
    - Drops multicollinear columns: 'Aerosol_OD_Min', 'Aerosol_OD_Max', 'Aerosol_OD_Std'.
    - Creates a 'season' column based on 'month_number'.
    - Drops temporal columns: 'month_number' and 'year'.
    """
    # Remove outliers based on the "Aerosol_OD_Mean" column
    lower_quantile = df["Aerosol_OD_Mean"].quantile(0.1)
    upper_quantile = df["Aerosol_OD_Mean"].quantile(0.9)
    
    df = df[
        (df["Aerosol_OD_Mean"] >= lower_quantile) & 
        (df["Aerosol_OD_Mean"] <= upper_quantile)
    ].copy()  # Create a deep copy to avoid warnings
    
    # Remove multicollinear columns
    df.drop(columns=["Aerosol_OD_Min", "Aerosol_OD_Max", "Aerosol_OD_Std"], inplace=True)
    
    # Add a 'season' column based on month_number
    df.loc[:, 'season'] = df['month_number'].apply(lambda month: 
        'Winter' if month in [12, 1, 2] else 
        'Spring' if month in [3, 4, 5] else 
        'Summer' if month in [6, 7, 8] else 
        'Fall'
    )
    
    # Drop temporal columns
    df.drop(columns=["month_number", "year"], inplace=True)
    
    return df

@task
def split_dataset(df: pd.DataFrame):
    """
    Splits the DataFrame into training and testing sets.
    - X: Features (all columns except 'Aerosol_OD_Mean')
    - y: Target ('Aerosol_OD_Mean')
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    # Separate features and target
    X = df.drop(columns=["Aerosol_OD_Mean"])
    y = df["Aerosol_OD_Mean"]

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return [X_train, X_test, y_train, y_test]


@task
def model(X_train, y_train):
    # Build Model
    model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(strategy="mean"),
    Ridge() 
    )
    
    # Fit model

    model.fit(X_train, y_train)
    
    print(f"Model score: {model.score(X_train, y_train):.4f}")

    return model


@task
def communicate_result(model):
    # Extract coefficients from the model
    coefficients = model.named_steps["ridge"].coef_  # Access the Ridge model's coefficients

    # Get feature names
    features = model.named_steps["onehotencoder"].get_feature_names_out()  # Access the encoder's feature names

    # Create the Series
    feat_imp = pd.Series(coefficients, index=features)

    # Sort by absolute value
    feat_imp = feat_imp.reindex(feat_imp.abs().sort_values().index)

    # Prepare the formula string
    formula = "y = "

    # Iterate over features and coefficients to build the formula dynamically
    for feature, coef in zip(feat_imp.index, feat_imp):
        if coef >= 0:
            formula += f" + ({coef:.4f} * {feature})"
        else:
            formula += f" - ({abs(coef):.4f} * {feature})"

    print("Model Formula:", formula)

    # Create a DataFrame for easy interpretation
    feat_imp_df = feat_imp.reset_index()
    feat_imp_df.columns = ["Feature", "Coefficient"]

    print("Feature Importance:")
    print(feat_imp_df)
    return feat_imp_df




@task
def evaluate_model(y_test, y_test_pred):
    """
    Evaluates the model performance using multiple metrics.
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² (Coefficient of Determination)
    
    Parameters:
    - y_test: Actual target values in the test set
    - y_test_pred: Predicted target values by the model
    
    Returns:
    - A dictionary with evaluation metrics
    """
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    r2 = r2_score(y_test, y_test_pred)
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return {"MAE": mae, "RMSE": rmse, "R²": r2}

@task
def plot_feature_importance(model):
    """
    Plots the feature importances (coefficients) from the model.
    
    Parameters:
    - model: The trained model
    
    Displays:
    - A horizontal bar plot of feature importances.
    """
    # Extract coefficients from the model
    coefficients = model.named_steps["ridge"].coef_
    features = model.named_steps["onehotencoder"].get_feature_names_out()
    
    # Create a Series for feature importances
    feat_imp = pd.Series(coefficients, index=features)
    
    # Sort by absolute value
    feat_imp = feat_imp.reindex(feat_imp.abs().sort_values().index)
    
    # Plot the feature importances
    feat_imp.plot(kind="barh", figsize=(10, 6))
    plt.title("Feature Importance (Coefficients from Ridge Model)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Features")
        
@task
def plot_residuals(y_test, y_test_pred):
    """
    Plots the residuals distribution to visualize how well the model fits the data.
    
    Parameters:
    - y_test: Actual target values in the test set
    - y_test_pred: Predicted target values by the model
    
    Displays:
    - A histogram with a kernel density estimate (KDE) of residuals.
    """
    residuals = y_test - y_test_pred
    sns.histplot(residuals, kde=True, bins=30, color='blue')
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


@flow
def model_flow():
    logger = get_run_logger()
    logger.info("Starting the model flow...")

    # Run the ingestion flow to get the renamed data file
    logger.info("Running the ingestion flow to retrieve the renamed data file...")
    renamed_data_path = main_flow()
    
    # Load and clean the data
    logger.info(f"Loading and cleaning data from {renamed_data_path}...")
    renamed_df = create_dataframe(renamed_data_path)
    cleaned_df = wrangle(renamed_df)
    
    # Split the dataset into training and testing sets
    logger.info("Splitting the dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = split_dataset(cleaned_df)
    
    # Train the model
    logger.info("Training the model...")
    model_pipeline = model(X_train, y_train)
    
    # Make predictions
    logger.info("Making predictions on the test set...")
    y_test_pred = model_pipeline.predict(X_test)
    
    # Evaluate the model
    logger.info("Evaluating the model...")
    evaluate_model(y_test, y_test_pred)
    
    # Calculate and display baseline MAE
    y_mean = y_train.mean()
    y_pred_baseline = [y_mean] * len(y_train)
    baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
    logger.info(f"Aerosol_OD_Mean: {y_mean}")
    logger.info(f"Baseline MAE: {baseline_mae}")
    
    # Calculate and display test baseline MAE
    y_test_mean = y_test.mean()
    y_test_baseline = [y_test_mean] * len(y_test)
    test_baseline_mae = mean_absolute_error(y_test, y_test_baseline)
    logger.info(f"Aerosol_test_OD_Mean: {y_test_mean}")
    logger.info(f"Test_Baseline MAE: {test_baseline_mae}")
    
    # Plot feature importance
    logger.info("Plotting feature importance...")
    plot_feature_importance(model_pipeline)
    
    # Plot residuals
    logger.info("Plotting residuals...")
    plot_residuals(y_test, y_test_pred)
    
    plt.show()
    logger.info("Model flow completed.")

if __name__ == "__main__":
    model_flow()
