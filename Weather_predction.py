import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
file_path = "E:\\E Drive\\Engineering\\5thsem\\mini project\\3857413.csv"
data = pd.read_csv(file_path)

# Step 1: Handle Missing Data
data['PRCP'] = data['PRCP'].interpolate(method='linear').fillna(0)  # Interpolate and fill missing PRCP with 0
data['TAVG'] = data['TAVG'].interpolate(method='linear')
data['TMAX'] = data['TMAX'].interpolate(method='linear')
data['TMIN'] = data['TMIN'].interpolate(method='linear')

# Step 2: Feature Engineering
data['DATE'] = pd.to_datetime(data['DATE'])
data['Year'] = data['DATE'].dt.year
data['Month'] = data['DATE'].dt.month
data['Day'] = data['DATE'].dt.day
data['Temp_Range'] = data['TMAX'] - data['TMIN']

# Classify Average Temperature into categories: Hot, Warm, Cold
def classify_temperature(avg_temp):
    if avg_temp > 80:  # Hot
        return 'Hot'
    elif avg_temp > 60:  # Warm
        return 'Warm'
    else:  # Cold
        return 'Cold'

data['Temp_Category'] = data['TAVG'].apply(classify_temperature)

# Classify whether it will rain based on precipitation: 1 if rain (PRCP > 0), else 0
data['Rain'] = (data['PRCP'] > 0).astype(int)

# Step 3: Train Models for Historical Data
def split_and_train(data):
    # Features and targets
    X = data[['TMAX', 'TMIN', 'PRCP', 'Month', 'Temp_Range']]
    y_temp = data['Temp_Category']
    y_rain = data['Rain']
    
    # Split dataset
    X_train, X_test, y_train_temp, y_test_temp = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(X, y_rain, test_size=0.2, random_state=42)
    
    # Train models
    temp_classifier = RandomForestClassifier(random_state=42)
    temp_classifier.fit(X_train, y_train_temp)
    
    rain_classifier = RandomForestClassifier(random_state=42)
    rain_classifier.fit(X_train_rain, y_train_rain)
    
    # Calculate accuracy
    temp_accuracy = accuracy_score(y_test_temp, temp_classifier.predict(X_test))
    rain_accuracy = accuracy_score(y_test_rain, rain_classifier.predict(X_test_rain))

    return temp_classifier, rain_classifier, temp_accuracy, rain_accuracy

# Train the classifiers and get accuracy
temp_classifier, rain_classifier, temp_accuracy, rain_accuracy = split_and_train(data)

# Streamlit Web Interface
st.set_page_config(page_title="Weather Prediction App", page_icon="🌦️", layout="wide")
st.sidebar.title("Navigation")
selected = st.sidebar.radio("Go to", ["Home", "Prediction", "About"])

# Home Page
if selected == "Home":
    st.title("🌦️ Weather Prediction App")
    st.markdown("""  
        Welcome to the **Weather Prediction App**!  
        Use this app to predict:
        - **Temperature categories**: Hot, Warm, or Cold  
        - **Rain status**: Rain or No Rain  

        Navigate using the sidebar.  
        """)
    # st.markdown(f"### Model Accuracy")
    # st.metric(label="Temperature Prediction Accuracy", value=f"{temp_accuracy * 100:.2f}%")
    # st.metric(label="Rain Prediction Accuracy", value=f"{rain_accuracy * 100:.2f}%")

# Prediction Page
elif selected == "Prediction":
    st.title("🔮 Weather Prediction")
    st.markdown("Select a date to get weather predictions for temperature and precipitation.")

    # User selects a date (allow both past and future dates)
    user_date = st.date_input("Select a Date:", min_value=data['DATE'].min(), max_value=pd.to_datetime('today') + pd.DateOffset(365))  # Allow one year in the future

    # Add a button to trigger predictions
    if st.button("Show Prediction"):
        # Check if the selected date is a historical date or a future date
        is_future_date = pd.Timestamp(user_date) > pd.to_datetime('today')
        
        # Check if data for the selected date exists in historical data
        actual_row = data[data['DATE'] == pd.Timestamp(user_date)]
        
        if not actual_row.empty and not is_future_date:
            # Predict using historical data for an existing date
            features = actual_row[['TMAX', 'TMIN', 'PRCP', 'Month', 'Temp_Range']]
            predicted_temp_category = temp_classifier.predict(features)[0]
            predicted_rain = rain_classifier.predict(features)[0]
            predicted_avg_temp = (actual_row['TMAX'].values[0] + actual_row['TMIN'].values[0]) / 2
            predicted_prcp = actual_row['PRCP'].values[0]  # Using actual PRCP as a placeholder for prediction

            # Calculate prediction accuracy
            actual_avg_temp = actual_row['TAVG'].values[0]
            temp_diff = abs(predicted_avg_temp - actual_avg_temp)
            temp_accuracy = max(0, 100 - (temp_diff / actual_avg_temp) * 100) if actual_avg_temp != 0 else 100

            actual_rain = actual_row['Rain'].values[0]
            rain_accuracy = 100 if predicted_rain == actual_rain else 0

            # Display Results with Actual Data
            st.markdown("### Prediction Results (Based on Historical Data)")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Temperature Category", value=predicted_temp_category)
                st.metric(label="Predicted Rain", value="Rain" if predicted_rain else "No Rain")
                st.metric(label="Predicted Avg Temperature", value=f"{predicted_avg_temp:.2f}")
                st.metric(label="Predicted Precipitation", value=f"{predicted_prcp:.2f} inches")
            with col2:
                st.metric(label="Actual Temperature Category", value=actual_row['Temp_Category'].values[0])
                st.metric(label="Actual Rain", value="Rain" if actual_rain else "No Rain")
                st.metric(label="Actual Avg Temperature", value=f"{actual_avg_temp:.2f}")
                st.metric(label="Actual Precipitation", value=f"{actual_row['PRCP'].values[0]:.2f} inches")

            # Display Accuracy
            st.markdown("### Prediction Accuracy for Selected Date")
            st.metric(label="Temperature Prediction Accuracy", value=f"{temp_accuracy:.2f}%")
            st.metric(label="Rain Prediction Accuracy", value=f"{rain_accuracy:.2f}%")

        else:
            # For future date prediction (no actual data to compare)
            # Calculate averages from historical data
            avg_tmax = data['TMAX'].mean()
            avg_tmin = data['TMIN'].mean()
            avg_prcp = data['PRCP'].mean()
            temp_range_avg = avg_tmax - avg_tmin
            
            future_features = pd.DataFrame([[avg_tmax, avg_tmin, avg_prcp, user_date.month, temp_range_avg]], 
                                           columns=['TMAX', 'TMIN', 'PRCP', 'Month', 'Temp_Range'])

            # Predict temperature category and rain (using historical averages)
            predicted_temp_category = temp_classifier.predict(future_features)[0]
            predicted_rain = rain_classifier.predict(future_features)[0]
            predicted_avg_temp = (avg_tmax + avg_tmin) / 2
            predicted_prcp = avg_prcp
            
            # Display Results for Future Date
            st.markdown("### Future Prediction Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Predicted Temperature Category", value=predicted_temp_category)
                st.metric(label="Predicted Rain", value="Rain" if predicted_rain else "No Rain")
                st.metric(label="Predicted Avg Temperature", value=f"{predicted_avg_temp:.2f}")
                st.metric(label="Predicted Precipitation", value=f"{predicted_prcp:.2f} inches")
            with col2:
                st.markdown("**Note**: Predictions are based on historical averages.")

# About Page
elif selected == "About":
    st.title("📖 About")
    st.markdown(""" 
        This application uses historical weather data to predict:
        - Temperature categories (Hot, Warm, Cold)
        - Rain status (Rain or No Rain)
        - Precipitation values

        ### Tools Used:
        - **Python** for data processing
        - **Scikit-learn** for machine learning
        - **Streamlit** for the interactive web interface

        """)
