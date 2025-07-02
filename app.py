from flask import Flask, render_template, request
import numpy as np
import pickle
from alibi.explainers import AnchorTabular

app = Flask(__name__)

# Load model and data
model, data = pickle.load(open('models/model.pkl', 'rb'))

# Class names for prediction labels
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Initialize the Anchor explainer
explainer = AnchorTabular(model.predict, feature_names=data.feature_names)
explainer.fit(data.data, disc_perc=(25, 50, 75))

@app.route('/')
def index():
    return render_template('index.html', feature_names=data.feature_names)

@app.route('/explain', methods=['POST'])
def explain():
    try:
        input_data = [float(x) for x in request.form.getlist('feature')]
        instance = np.array(input_data).reshape(1, -1)
        prediction_index = model.predict(instance)[0]
        prediction_label = class_names[prediction_index]

        explanation = explainer.explain(instance[0])

        result = {
            "prediction": prediction_label,
            "anchor": explanation.anchor,
            "precision": round(explanation.precision, 2),
            "coverage": round(explanation.coverage, 2)
        }

        return render_template('index.html',
                               feature_names=data.feature_names,
                               result=result,
                               input_data=input_data)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    print("✅ Flask app running at http://127.0.0.1:5050/")
    app.run(host="0.0.0.0", port=5050, debug=True)


