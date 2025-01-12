from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
import numpy as np
import joblib

app = Flask(__name__)

session = ort.InferenceSession("model.onnx")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # data = request.get_json(force=True)

    try:
        # data_input = [data['rainfall_mm'],
        #               data['soil_quality_index'],
        #               data['farm_size_hectares'],
        #               data['sunlight_hours'],
        #               data['fertilizer_kg']]
        data_input = [
            request.form['rainfall_mm'],
            request.form['soil_quality_index'],
            request.form['farm_size_hectares'],
            request.form['sunlight_hours'],
            request.form['fertilizer_kg']
        ]
        data_input = np.array([data_input], dtype=np.float32)
        data_input_scaled = scaler.transform(data_input)
        input_name = session.get_inputs()[0].name

        pred_onnx = session.run(None, {input_name: data_input_scaled})

        prediction = pred_onnx[0].tolist()

        # return jsonify({'prediction': pred_onnx[0].tolist()}), 200
        return render_template("predict.html", prediction=prediction)

    except ValueError as ve:
        return jsonify("error: Input invalid"), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
