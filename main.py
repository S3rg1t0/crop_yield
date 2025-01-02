from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import joblib

app = Flask(__name__)

session = ort.InferenceSession("model.onnx")
labelEncoder1 = joblib.load("labelEncoder1.pkl")
labelEncoder3 = joblib.load("labelEncoder3.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        data_input = [data['rainfall_mm'],
                      int(labelEncoder1.transform([data['soil_quality_index']])[0]),
                      data['farm_size_hectares'],
                      int(labelEncoder3.transform([data['sunlight_hours']])[0]),
                      data['fertilizer_kg']]
        data_input = np.array([data_input], dtype=np.float32)
        input_name = session.get_inputs()[0].name

        pred_onnx = session.run(None, {input_name: data_input})

        return jsonify({'prediction': pred_onnx[0].tolist()}), 200

    except ValueError as ve:
        return jsonify("error: Input invalid"), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
