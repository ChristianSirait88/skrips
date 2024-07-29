import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# Load your data (replace with your actual data)
df = pd.read_excel('clean.xlsx')

df = df.drop(columns=['NAMAPENYAKIT'])
df = df.drop(columns=['HARGAPREMI'])

df['HARGAPREMI'] =np.ceil(df['BAYAR'] * (100 / 80))

unique_column = 'NAMAPERUSAHAAN' 
X=df.drop(['HARGAPREMI',],axis=1)
y=df['HARGAPREMI']
    
#split the dataset for train&test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

perusahaan = X_test['NAMAPERUSAHAAN']
jenis_klaim = X_test['JENISKLAIM']
X_train = X_train.drop(columns=['NAMAPERUSAHAAN'])
X_test = X_test.drop(columns=['NAMAPERUSAHAAN'])
X_train = X_train.drop(columns=['JENISKLAIM'])
X_test = X_test.drop(columns=['JENISKLAIM'])

def xgboostmodel(df):
    model = xgb.XGBRegressor(
        learning_rate=0.1,
        max_depth=10,
        n_estimators=100,
        colsample_bytree=1  
    )
    # Add 10-fold cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    print(f"Cross-validation MAE: {np.mean(-scores)}")

    model.fit(X_train, y_train)
    return model

df = df.drop(columns=['NAMAPERUSAHAAN'])
df = df.drop(columns=['JENISKLAIM'])

model = xgboostmodel(df)

# Evaluate on test data
predicted_prices = model.predict(X_test)
r2 = r2_score(y_test, predicted_prices)

def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

mape = calculate_mape(y_test, predicted_prices)
print(f"MAPE: {mape:.2f}%")
print(f"Test R-squared (r2): {r2:.4f}")
print(predicted_prices)


feature_importance =model.feature_importances_

X = X.drop(columns=['NAMAPERUSAHAAN'])
X = X.drop(columns=['JENISKLAIM'])

# Create a DataFrame to display feature importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)


X_test["HASIL"] = predicted_prices
X_test["NAMAPERUSAHAAN"] = perusahaan
X_test["JENISKLAIM"] = jenis_klaim

X_test.to_excel('hasil.xlsx', index=False)
