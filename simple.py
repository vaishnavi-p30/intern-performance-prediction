from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

print("🔄 Loading models...")

# Load model and scaler
try:
    model = joblib.load('optimized_random_forest_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return {
        "message": "Intern Performance Prediction API",
        "endpoints": {
            "GET /health": "Check API status",
            "POST /predict": "Make prediction",
            "GET /info": "Get model info"
        }
    }

@app.route('/health')
def health():
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None
    }

@app.route('/info')
def info():
    return {
        "model_type": "Random Forest Regressor",
        "features": ["Task_Completion", "Consistency", "Engagement"],
        "prediction_range": "0-100"
    }

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return {"error": "Model not loaded"}, 503
    
    try:
        # Get data from request
        data = request.get_json()
        
        task = float(data['Task_Completion'])
        cons = float(data['Consistency'])
        eng = float(data['Engagement'])
        
        # Create engineered features (same as training)
        total = task + cons + eng
        avg = total / 3
        ratio = task / (cons + 0.01)
        diff = eng - cons
        var = np.var([task, cons, eng])
        
        # Create features array
        features = np.array([[task, cons, eng, total, avg, ratio, diff, var]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        return {
            "predicted_performance": round(prediction, 2),
            "input": {
                "Task_Completion": task,
                "Consistency": cons,
                "Engagement": eng
            }
        }
        
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    print("\n🚀 Starting API server...")
    print("📍 URL: http://localhost:8000")
    print("📚 Test with: python test_simple.py")
    app.run(host='0.0.0.0', port=8000, debug=False)