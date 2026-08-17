import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("="*10)
print("STEP 1: LOADING DATA")
print("20"*10)
# loading the housing dataset
housing = fetch_california_housing()

# x --> features (what the model looks at)
x = housing.data
# y --> target (what we are trying to predict)
y= housing.target

print(f"Data loaded succesfully!")
print(f" Number of houses: {len(x)}")
print(f" Number of features: {x.shape[1]}")
print(f" Feature names: {housing.feature_names}")
print(f" Target names: {housing.target_names[0]}")
"""
here the model looks at the features(X=  a rows * b columns)
while y = a prices
"""

# Here we explore the data

print("\n" + "="*10)
print("STEP 2: EXPLORE THE DATA(STATISTICS)")
print("="*10)

# convert to pandas dataframe(for visuals)
df = pd.DataFrame(x, columns=housing.feature_names)
df['Price'] = y #add the target

print("First 5 rows of data:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe()) # mean, std, min, max for each column

print("\nCorrelations with price")
correlations = df.corr()['Price'].sort_values(ascending=False)
print(correlations)

# describe() gives you the mean, standard deviation, min, max, percentiles
# corr() shows which features are most related to price

# Here we split the data

print("\n" + "="*10)
print("STEP 3: SPLIT INTO TRAIN AND TEST")
print("="*50)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2, # 20% for testing
    random_state=42 # For reproducable results
)

print(f"Data split successfully")
print(f" Training set: {len(x_train)} houses (80%)")
print(f" Testing set: {len(x_test)} houses (20%)")

"""
the model trains on 80% of the data then tests on 20% of the data
the 'random_state=42' makes sure we get the same split everytime
"""

# Here we create and train the model

print("\n" + "="*10)
print("STEP 4: BUILD AND TRAIN THE ML MODEL")
print("="*10)

#create the model (this is the brain)
model = LinearRegression()

print("Model created: Linear Regression")
print(" This will find the best mathematical formula:")
print(" Price = b0 + b1*Income + b2*age + b3*rooms+...")

#Train the model (this is where learning happens)
model.fit(x_train, y_train)

print("Model trained succesfully")
print(" The model has learned the optimal coefficients:")

# Show what the model learned (the coefficients)
for name, coef in zip(housing.feature_names, model.coef_):
    print(f"  {name}: {coef:.4f}")

print(f"  Intercept(b0): {model.intercept_:.4f}")

"""
here, the model found the best coefficients
these minimize the error btwn predictions and actual prices
this is a multiple  linear regression
"""

#here we make predictions

print("\n" + "="*10)
print("STEP 5: MAKE PREDICTIONS")
print("="*10)

#predict on the test data (houses the model has never seen)
y_pred = model.predict(x_test)

#convert from $100,000s to actual dollars
y_test_dollars = y_test * 100000
y_pred_dollars = y_pred * 100000

print("Prediction made!")
print(f" First 5 actual prices: {y_test_dollars[:5].astype(int)}")
print(f" First 5 predicted prices: {y_pred_dollars[:5].astype(int)}")

#calculate errors
errors = y_pred_dollars - y_test_dollars
print(f"\n First 5 erors: {errors[:5].astype(int)}")
print(f"  (Positive = overpredicted, Negative = underpredicted)")

# Here we will evaluate the model

print("\n" + "="*10)
print("STEP 6: EVALUATE PERFOMANCE")
print("="*10)

#calculate metrics

rmse = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
r2 = r2_score(y_test_dollars, y_pred_dollars)
mae = np.mean(np.abs(y_pred_dollars - y_test_dollars))

print("   Model Performance:")
print(f"   RMSE: ${rmse:,.0f}")
print(f"   MAE:  ${mae:,.0f}")
print(f"   R²:   {r2:.4f}")

print("\n Interpretation:")
print(f"   • On average, we're off by ${mae:,.0f}")
print(f"   • The model explains {r2*100:.1f}% of price variation")
print(f"   • Typical error: ${rmse:,.0f}")

# What just happened?
# RMSE = Root Mean Squared Error (average error, larger errors count more)
# MAE = Mean Absolute Error (average absolute error)
# R² = Coefficient of determination (you know this from statistics!)

# Visualization of the results

print("\n" + "="*10)
print("STEP 7: VISUALIZE THE RESULTS")
print("="*10)

#create a scatter plot: actual vs predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test_dollars, y_pred_dollars, alpha=0.5, s=10)
plt.plot([y_test_dollars.min(), y_test_dollars.max()],
         [y_test_dollars.min(), y_test_dollars.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)

#save the plot
plt.savefig('model_perfomance.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'model_perfomance.png'")
plt.show()

#here we predict a new house
print("\n" + "="*10)
print("STEP 8: PREDICT A NEW HOUSE")
print("="*10)

#lets create a hypothetical house
new_house = np.array([[
    5.0,   # MedInc: median income in area ($50,000)
    25.0,  # HouseAge: 25 years old
    6.0,   # AveRooms: 6 rooms average
    1.5,   # AveBedrms: 1.5 bedrooms average
    800,   # Population: 800 people in area
    3.0,   # AveOccup: 3 people per household
    34.0,  # Latitude: 34° North
    -118.0 # Longitude: -118° West
    ]])

#predict price
predicted_price = model.predict(new_house)[0]*100000

print("   New House Details:")
print(f"   Income area: ${new_house[0][0] * 10000:,.0f}")
print(f"   House age: {new_house[0][1]:.0f} years")
print(f"   Rooms: {new_house[0][2]:.1f}")
print(f"   Bedrooms: {new_house[0][3]:.1f}")
print(f"   Population: {new_house[0][4]:.0f}")
print(f"   Occupancy: {new_house[0][5]:.1f} people")
print(f"   Location: ({new_house[0][6]:.2f}, {new_house[0][7]:.2f})")

print(f"\n  Predicted Price: ${predicted_price:,.0f}")

# Summary

print("\n" + "=" * 50)
print(" PROJECT COMPLETE!")
print("=" * 50)
print("\nWhat We Accomplished:")
print("   1. Loaded California housing data")
print("   2. Explored the data (statistics!)")
print("   3. Split into train/test sets")
print("   4. Built a Linear Regression model")
print("   5. Trained the model on 80% of data")
print("   6. Made predictions on 20% of data")
print("   7. Evaluated performance (RMSE, R²)")
print("   8. Visualized results")
print("   9. Predicted price for a new house")
print("\n You just built your first ML model!")