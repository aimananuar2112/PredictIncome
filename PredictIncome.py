import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and preprocess the data
df = pd.read_csv("adult.csv")
df.columns = df.columns.str.replace(" ", "")
categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Copy the original DataFrame to avoid modifying it directly
df_encoded = df.copy()

# Initialize the LabelEncoder
labelencoder = LabelEncoder()

# Apply LabelEncoder to each categorical column
for column in categorical_columns:
    df_encoded[column] = labelencoder.fit_transform(df_encoded[column])

# Define y and X using the encoded DataFrame
y = df_encoded["income"]
X = df_encoded[['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Fit the scaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Create Streamlit app
st.write("""
# Income Prediction App
This app predicts whether an individual's income is more or less than $50K based on certain features.
""")

st.sidebar.header('User Input Parameters')

# Function to get user input from the sidebar
def user_input_features():
    age = st.sidebar.slider('Age', int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    workclass = st.sidebar.selectbox('Workclass', df['workclass'].unique())
    fnlwgt = st.sidebar.number_input('FNLWGT', int(df['fnlwgt'].min()), int(df['fnlwgt'].max()), int(df['fnlwgt'].mean()))
    education = st.sidebar.selectbox('Education', df['education'].unique())
    education_num = st.sidebar.slider('Education Num', int(df['education.num'].min()), int(df['education.num'].max()), int(df['education.num'].mean()))
    marital_status = st.sidebar.selectbox('Marital Status', df['marital.status'].unique())
    occupation = st.sidebar.selectbox('Occupation', df['occupation'].unique())
    relationship = st.sidebar.selectbox('Relationship', df['relationship'].unique())
    race = st.sidebar.selectbox('Race', df['race'].unique())
    sex = st.sidebar.selectbox('Sex', df['sex'].unique())
    capital_gain = st.sidebar.number_input('Capital Gain', int(df['capital.gain'].min()), int(df['capital.gain'].max()), int(df['capital.gain'].mean()))
    capital_loss = st.sidebar.number_input('Capital Loss', int(df['capital.loss'].min()), int(df['capital.loss'].max()), int(df['capital.loss'].mean()))
    hours_per_week = st.sidebar.slider('Hours per Week', int(df['hours.per.week'].min()), int(df['hours.per.week'].max()), int(df['hours.per.week'].mean()))
    native_country = st.sidebar.selectbox('Native Country', df['native.country'].unique())

    data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education.num': education_num,
        'marital.status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
        'hours.per.week': hours_per_week,
        'native.country': native_country
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Encode the categorical features
input_encoded = input_df.copy()
for column in categorical_columns:
    input_encoded[column] = labelencoder.fit_transform(input_encoded[column])

# Scale the numerical features
input_scaled = scaler.transform(input_encoded)

# Predict the class
prediction = model.predict(input_scaled)

# Display user input and prediction
st.subheader('User Input parameters')
st.write(input_df)

st.subheader('Prediction')
income_class = '>50K' if prediction[0] == 1 else '<=50K'
st.write(f'The predicted income class is: {income_class}')
