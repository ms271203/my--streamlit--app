import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data and model (simple linear regression example)
data = {'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Marks': [35, 45, 50, 55, 60, 70, 75, 85, 90]
        }

df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Marks']

# Create and train a simple model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title('Student Marks Prediction')
st.write('Enter hours studied and get predicted marks.')

# User input
hours = st.number_input('Enter Hours Studied:', min_value=0, max_value=10, value=1)
prediction = model.predict([[hours]])

# Display the prediction
st.write(f'Predicted Marks for {hours} hours of study: {prediction[0]:.2f}')

