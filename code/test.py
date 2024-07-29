import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load your data (replace with your actual data)
df = pd.read_excel('clean.xlsx')

# Drop irrelevant columns
df = df.drop(columns=['NAMAPENYAKIT', 'HARGAPREMI'])

# Calculate the target variable ('HARGAPREMI')
df['HARGAPREMI'] = np.ceil(df['BAYAR'] * (100 / 80))

# Extract relevant columns
unique_column = 'NAMAPERUSAHAAN'
X = df.drop(['HARGAPREMI'], axis=1)
y = df['HARGAPREMI']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Extract additional columns for later use
perusahaan = X_test['NAMAPERUSAHAAN']
jenis_klaim = X_test['JENISKLAIM']
X_train = X_train.drop(columns=['NAMAPERUSAHAAN', 'JENISKLAIM'])
X_test = X_test.drop(columns=['NAMAPERUSAHAAN', 'JENISKLAIM'])

def xgboost_model(df):
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=10,
        n_estimators=100
    )
    # Add 10-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    print(f"Cross-validation MAE: {np.mean(-scores)}")

    model.fit(X_train, y_train)
    return model

# Train the XGBoost model
model = xgboost_model(df)

# Evaluate on test data
predicted_prices = model.predict(X_test)
r2 = r2_score(y_test, predicted_prices)

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape = calculate_mape(y_test, predicted_prices)
print(f"MAPE: {mape:.2f}%")
print(f"Test R-squared (r2): {r2:.4f}")
print(predicted_prices)

# Extract feature importance scores
feature_importance = model.feature_importances_
# Create a DataFrame to display feature importance scores
feature_importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

def summarize_feature_importance(feature_importance_df):
    """
    Summarizes feature importance scores and ranks features.
    
    Args:
        feature_importance_df (pd.DataFrame): DataFrame containing feature importance scores.
    
    Returns:
        pd.DataFrame: Summary DataFrame with ranked features.
    """
    # Sort features by importance (descending order)
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Add a rank column
    feature_importance_df['Rank'] = range(1, len(feature_importance_df) + 1)
    
    return feature_importance_df

# Call the function to get the summary
summary_df = summarize_feature_importance(feature_importance_df)

# Display the summary
print(summary_df)

# Add predicted prices, company names, and claim types back to the test data
X_test["HASIL"] = predicted_prices
X_test["NAMAPERUSAHAAN"] = perusahaan
X_test["JENISKLAIM"] = jenis_klaim

# Save the modified test data (including predictions) to an Excel file
X_test.to_excel('hasil.xlsx', index=False)
