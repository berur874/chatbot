# ==========================================
# COMPARE LINEAR REGRESSION VS RANDOM FOREST
# ==========================================

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import joblib

# 1. Load data
print("Loading California housing data...")
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

print(f"✅ Data loaded: {X.shape[0]} houses, {X.shape[1]} features")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {len(X_train)} houses")
print(f"✅ Test set: {len(X_test)} houses")

# 3. Train Linear Regression (Our old model)
print("\n🔵 Training Linear Regression...")
start = time.time()
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_time = time.time() - start
lr_pred = lr.predict(X_test)

# 4. Train Random Forest (New model!)
print("🟢 Training Random Forest...")
start = time.time()
rf = RandomForestRegressor(
    n_estimators=100,  # Number of trees
    random_state=42
)
rf.fit(X_train, y_train)
rf_time = time.time() - start
rf_pred = rf.predict(X_test)

# 5. Compare Results
print("\n" + "=" * 50)
print("📊 MODEL COMPARISON")
print("=" * 50)

# Convert to dollars
y_test_dollars = y_test * 100000

# Linear Regression
lr_rmse = np.sqrt(mean_squared_error(y_test_dollars, lr_pred * 100000))
lr_r2 = r2_score(y_test, lr_pred)

# Random Forest
rf_rmse = np.sqrt(mean_squared_error(y_test_dollars, rf_pred * 100000))
rf_r2 = r2_score(y_test, rf_pred)

print(f"\n🔵 LINEAR REGRESSION:")
print(f"   RMSE: ${lr_rmse:,.0f}")
print(f"   R²:   {lr_r2:.4f}")
print(f"   Time: {lr_time:.2f} seconds")

print(f"\n🟢 RANDOM FOREST:")
print(f"   RMSE: ${rf_rmse:,.0f}")
print(f"   R²:   {rf_r2:.4f}")
print(f"   Time: {rf_time:.2f} seconds")

print(f"\n📈 IMPROVEMENT:")
print(f"   RMSE reduced by: ${(lr_rmse - rf_rmse):,.0f} ({(1 - rf_rmse/lr_rmse)*100:.1f}%)")
print(f"   R² improved by:  {(rf_r2 - lr_r2)*100:.1f} percentage points")

# 6. Feature Importance (This is where Random Forest shines!)
print("\n" + "=" * 50)
print("🔑 FEATURE IMPORTANCE")
print("=" * 50)

importances = rf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"{row['Feature']:12s} {bar} {row['Importance']:.2%}")

joblib.dump(rf, 'house_price_model.pk')
print("Model saved as 'house_price_model.pk'")
