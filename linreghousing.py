import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/Users/aaravkapadia/Linear Regression Housing/housing.csv")
#print(df.info())
df = df.fillna(2)
# # Summary statistics of the dataset
# print(df.describe())
# # Check for missing values
# print(df.isnull().sum())
#X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'price']]
#X = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']]
#X = df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']]
#X = df[['housing_median_age', 'total_bedrooms', 'households', 'median_income']]
#X = df[['housing_median_age', 'total_rooms', 'population', 'median_income']]
#X = df[['housing_median_age', 'total_bedrooms', 'population', 'median_income']]
X = df[['housing_median_age', 'total_bedrooms', 'median_income', 'households', 'population']] #best possible combination of the factors.
#print(df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']].mean())
#data explanation:
#3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
#4. totalRooms: Total number of rooms within a block
#5. totalBedrooms: Total number of bedrooms within a block
#6. population: Total number of people residing within a block
#7. households: Total number of households, a group of people residing within a home unit, for a block
#8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)

y = df[['median_house_value']]
#medianHouseValue: Median house value for households within a block (measured in US Dollars)
#y = df[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Building the Linear Regression Model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)
# Model Evaluation
y_pred = model.predict(X_test)

#R-squared for model evaluation
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)
#Predictions vs acutal prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()

#Residual Plot
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Actual Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

#new_data = [[30, 2500, 525, 1600, 500, 4]]
#predicted_price = model.predict(new_data)


#print("Predicted Price:", predicted_price[0])
#print(y.mean())
