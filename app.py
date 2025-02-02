import logging
from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder and load any previously saved encoder if needed (if you saved it)
le_team = LabelEncoder()
le_toss = LabelEncoder()
le_city = LabelEncoder()

# You could also save the LabelEncoders after training and load them in the Flask app.
# e.g., le_team = joblib.load('team_encoder.joblib')
# You can use the same for 'team1', 'team2', 'toss_winner', 'toss_decision', 'city'


app = Flask(__name__)

# Enable logging for Flask to debug
app.logger.setLevel(logging.DEBUG)

# Load the model
model = joblib.load('ipl_score_prediction_model.joblib')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    app.logger.debug("Received request for prediction")
    
    # Ensure the request contains JSON data
    data = request.get_json()

    if not data:
        app.logger.error("No data received")
        return jsonify({"error": "No data received"}), 400  # Return error if no data is received
    
    try:
        app.logger.debug(f"Received data: {data}")

        # Encode categorical features (team1, team2, toss_winner, toss_decision, city)
        team1_encoded = le_team.fit_transform([data['team1']])[0]
        team2_encoded = le_team.fit_transform([data['team2']])[0]
        toss_winner_encoded = le_toss.fit_transform([data['toss_winner']])[0]
        toss_decision_encoded = le_toss.fit_transform([data['toss_decision']])[0]
        city_encoded = le_city.fit_transform([data['city']])[0]

        # Prepare the feature list for prediction
        features = [
            team1_encoded,
            team2_encoded,
            toss_winner_encoded,
            toss_decision_encoded,
            city_encoded,
            data['powerplay_runs'],
            data['death_overs_runs']
        ]
        
        # Make the prediction
        prediction = model.predict([features])
        app.logger.debug(f"Prediction made: {prediction[0]}")
        
        # Return the prediction as a valid JSON response
        return jsonify({'predicted_score': prediction[0]})
    
    except KeyError as e:
        app.logger.error(f'Missing field: {str(e)}')
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        app.logger.error(f'Prediction failed: {str(e)}')
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

        
if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)  # Ensure 'debug=True'

