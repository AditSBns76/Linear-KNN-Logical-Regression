import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Creating the DataFrame
data = {
    'Name': ['Ayush', 'Raghav', 'Yashvi', 'Aman', 'Nikhil', 'Yuvraj', 'Nischay'],
    'Age': [21, 18, 20, 16, 21, 20, 19],
    'Address': ['Delhi', 'Patna', 'Allahabad', 'Varanasi', 'Indore', 'Mumbai', 'Patiala'],
    'Qualification': ['Msc', 'MBBS', 'MCA', 'Phd', 'Btech', 'Mtech', 'BBA'],
    'Salaries': [85000, 70000, 65000, 100000, 50000, 200000, 90000],
}

df = pd.DataFrame(data)

# Creating a binary target variable based on Salaries
threshold = 100000
df['Salary_Category'] = np.where(df['Salaries'] > threshold, 1, 0)  # 1 for High, 0 for Low

# Selecting features and target variable
X = df[['Age']]  # Predictor variable
y = df['Salary_Category']  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the feature
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the logistic regression model
logistic_model = LogisticRegression()

# Fitting the model
logistic_model.fit(X_train, y_train)

# Making predictions
y_pred = logistic_model.predict(X_test)

# Evaluating the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting the logistic regression curve
plt.figure(figsize=(10, 6))

# Generate a range of values for Age
age_range = np.linspace(X['Age'].min(), X['Age'].max(), 300).reshape(-1, 1)
age_range_scaled = scaler.transform(age_range)

# Predict probabilities for the age range
probabilities = logistic_model.predict_proba(age_range_scaled)[:, 1]

# Plotting the probabilities
plt.plot(age_range, probabilities, color='green', label='Logistic Regression Curve')

# Scatter plot of actual data
plt.scatter(X['Age'], y, color='blue', label='Actual Categories')  # Actual data points
plt.scatter(X_test, y_pred, color='red', label='Predicted Categories')  # Predicted data points

plt.title('Logistic Regression: Age vs Salary Category')
plt.xlabel('Age')
plt.ylabel('Probability of High Salary (1)')
plt.xticks([16, 18, 19, 20, 21])
plt.yticks([0, 1])
plt.legend()
plt.grid()
plt.show()