from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset and model setup
data = pd.read_csv(r'food_data.csv')

# Define essential columns for input (minimized list for prediction)
essential_columns = ['Energy_kcal', 'Protein_g', 'Fat_g', 'Carb_g', 'VitC_mg', 'Iron_mg']

# Preprocessing: Filling missing values with the mean
data[essential_columns] = data[essential_columns].fillna(data[essential_columns].mean())

# Define the target columns and thresholds for deficiencies
deficiency_thresholds = {
    'VitA_mcg': 700, 'VitB6_mg': 1.3, 'VitB12_mcg': 2.4, 'VitC_mg': 75, 'Iron_mg': 18, 'Calcium_mg': 1000
}

# Creating binary deficiency targets
for nutrient, threshold in deficiency_thresholds.items():
    data[f'{nutrient}_Deficiency'] = np.where(data[nutrient] < threshold, 1, 0)

# Feature scaling and model setup
X = data[essential_columns]
scaler = StandardScaler().fit(X)
rf_clf = RandomForestClassifier(random_state=42).fit(scaler.transform(X), data[[f'{k}_Deficiency' for k in deficiency_thresholds]])

# Route for home page
@app.route('/')
def home():
    # Render a simpler HTML form with minimized essential inputs
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nutrition Deficiency Predictor</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    </head>
    <body  style ="background-color:orange">
        <div class="container">
            <h1 class="mt-5 text-center">Nutrition Deficiency Predictor</h1>
            <form id="foodForm" class="mt-3">
                {% for col in columns %}
                <div class="form-group">
                    <label for="{{ col }}">{{ col.replace('_', ' ') }}</label>
                    <input type="number" step="any" class="form-control" id="{{ col }}" placeholder="Enter value for {{ col.replace('_', ' ') }}">
                </div>
                {% endfor %}
                <button type="submit" class="btn btn-primary btn-block">Submit</button>
            </form>
            <div id="result" class="mt-4"></div>
        </div>

        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            $(document).ready(function() {
                $('#foodForm').on('submit', function(event) {
                    event.preventDefault();
                    let formData = {};
                    {% for col in columns %}
                    formData["{{ col }}"] = parseFloat($('#{{ col }}').val()) || 0;
                    {% endfor %}

                    $.ajax({
                        url: '/predict',
                        method: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify(formData),
                        success: function(response) {
                            let resultHtml = "<h3>Deficiency Results:</h3><ul>";
                            for (const [nutrient, deficiency] of Object.entries(response.deficiencies)) {
                                resultHtml += `<li>${nutrient}: ${deficiency ? "Deficient" : "Sufficient"}</li>`;
                            }
                            resultHtml += "</ul><h3>Recommendations:</h3><ul>";
                            for (const recommendation of response.recommendations) {
                                resultHtml += `<li>${recommendation}</li>`;
                            }
                            resultHtml += "</ul>";
                            $('#result').html(resultHtml);
                        },
                        error: function(error) {
                            $('#result').html('<p class="text-danger">Error processing the request. Please try again.</p>');
                        }
                    });
                });
            });
        </script>
    </body>
    </html>
    '''
    return render_template_string(html_content, columns=essential_columns)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    food_data = [input_data.get(col, 0) for col in essential_columns]

    # Scale the input data
    scaled_input = scaler.transform([food_data])

    # Make predictions for all deficiencies
    predictions = rf_clf.predict(scaled_input)

    deficiencies = {f"{nutrient}_Deficiency": bool(pred) for nutrient, pred in zip(deficiency_thresholds, predictions[0])}
    recommendations = [
        f"Increase intake of {nutrient.replace('_', ' ')}" for nutrient, pred in deficiencies.items() if pred
    ]

    return jsonify({
        'deficiencies': deficiencies,
        'recommendations': recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
