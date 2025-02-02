# streamlit_app.py (Streamlit UI)

import streamlit as st
import requests  # To make HTTP requests to the Flask API

# Streamlit UI components
st.title("IPL Score Prediction")

# Select team names, toss winner, and other inputs
team1 = st.selectbox("Select Team 1", ["Team A", "Team B", "Team C", "Team D"])
team2 = st.selectbox("Select Team 2", ["Team A", "Team B", "Team C", "Team D"])
toss_winner = st.selectbox("Select Toss Winner", ["Team A", "Team B", "Team C", "Team D"])
toss_decision = st.radio("Toss Decision", ["Bat", "Bowl"])
city = st.selectbox("Select City", ["City 1", "City 2", "City 3"])
powerplay_runs = st.number_input("Powerplay Runs", min_value=0)
death_overs_runs = st.number_input("Death Overs Runs", min_value=0)

# Make prediction when button is clicked
# Make prediction when button is clicked
if st.button("Predict Score"):
    # Create the feature data to send to the Flask API
    features = {
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'city': city,
        'powerplay_runs': powerplay_runs,
        'death_overs_runs': death_overs_runs
    }

    # Print features to verify correct data is being sent
    st.write("Sending the following data to Flask:")
    st.write(features)

    # Send POST request to Flask API
    try:
        response = requests.post("http://127.0.0.1:5001/predict", json=features)

        # Check if the response is valid
        if response.status_code == 200:
            try:
                prediction = response.json()['predicted_score']
                st.write(f"Predicted Score: {prediction}")
            except ValueError:
                st.write("Invalid response from Flask. Could not parse JSON.")
        else:
            st.write(f"Error from Flask: {response.json()['error']}")
    except requests.exceptions.RequestException as e:
        st.write(f"Error while connecting to Flask API: {e}")
