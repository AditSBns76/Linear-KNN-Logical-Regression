import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating the DataFrame
data = {
    'Name': ['Ayush', 'Raghav', 'Yashvi', 'Aman', 'Nikhil', 'Yuvraj', 'Nischay'],
    'Age': [21, 18, 20, 16, 21, 20, 19],
    'Address': ['Delhi', 'Patna', 'Allahabad', 'Varanasi', 'Indore', 'Mumbai', 'Patiala'],
    'Qualification': ['Msc', 'MBBS', 'MCA', 'Phd', 'Btech', 'Mtech', 'BBA'],
    'Salaries': [85000, 70000, 65000, 100000, 50000, 200000, 90000],
}

df = pd.DataFrame(data)

# Selecting features and target variable
X = df[['Age']]  # Predictor variable
y = df['Salaries']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Salaries:", y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Optional: Displaying the coefficients
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Salaries')  # Actual data points
plt.scatter(X_test, y_pred, color='red', label='Predicted Salaries')  # Predicted data points
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')  # Regression line
plt.title('Linear Regression: Age vs Salaries')
plt.xlabel('Age')
plt.ylabel('Salaries')
plt.legend()
plt.grid()
plt.show()