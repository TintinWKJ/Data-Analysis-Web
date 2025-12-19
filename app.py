import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from numpy.random import default_rng as rng

CSV_FILE = "sustainable_waste_management_dataset_2024.csv"
df = pd.read_csv(CSV_FILE)
st.dataframe(df)


selected_features = ['recyclable_kg',	'organic_kg','temp_c','rain_mm','collection_capacity_kg']
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=30)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))


plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Waste_kg (Y_test)')
plt.ylabel('Predicted Waste_kg (Y_pred)')
plt.title('Predicted Waste vs. Actual Waste')
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(fig)
