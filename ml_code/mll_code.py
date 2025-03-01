# Importing Libraries
import pandas as pd  # Dataframe Manipulation  
import numpy as np  # Array/lists Handlings
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns  # For data visualization
from pandas.api.types import is_numeric_dtype
import joblib 

# Importing Dataset
df = pd.read_csv("Water data.csv")

# Data Information
df.info()

# Describing the dataset
df.describe()
df.isna().sum()
df.dtypes

# Label Encoding
from sklearn.preprocessing import LabelEncoder

# Initialize encoder
encoder = LabelEncoder()

# Apply encoding
df["Source"] = encoder.fit_transform(df["Source"])  
df["Color"] = encoder.fit_transform(df["Color"])  
df["Month"] = encoder.fit_transform(df["Month"]) 

# Check unique values after encoding
print(df["Source"].unique())
print(df["Month"].unique())
print(df["Color"].unique())

# Handling Missing values
df.isna().sum()

# Data Cleaning
numeric_columns = []
for i in df.columns:
    if is_numeric_dtype(df[i]):
        numeric_columns.append(i)
print(numeric_columns)

numeric_columns = []
for i in df.columns:
    if is_numeric_dtype(df[i]):
        numeric_columns.append(i)

for i in numeric_columns:
    if -0.5 < df[i].skew() < 0.5:
        df.fillna(df[i].mean(), inplace=True)
    else:
        df.fillna(df[i].median(), inplace=True)

print(numeric_columns)
df.isna().sum()
df.head()

# Dataset splitting
X = df.iloc[:, :-1].values  # Dependent variable
y = df.iloc[:, -1].values   # Independent Vars

# Splitting into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Training the RandomForestRegressor model
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=10, random_state=42)
reg.fit(X_train, y_train)

# Making predictions
y_pred = reg.predict(X_test)

# Compare the predicted values with actual values
list1 = []
list1 = np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1)
for i in range(20):
    print(list1[i])

# Mean Absolute Error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
rsc = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("mean squared error: ", mse)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("R2 Score:", rsc)


# Saving the trained model
joblib.dump(reg, 'water_quality_model.pkl')

#Saving the label encoder as well (for encoding the categorical data when making predictions)
joblib.dump(encoder, 'label_encoder.pkl')