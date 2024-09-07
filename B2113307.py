from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy giá trị
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    features = [sepal_length, sepal_width, petal_length, petal_width]
    final_features = scaler.transform([features])
    
    prediction = model.predict(final_features)
    species = label_encoder.inverse_transform(prediction)[0]
    
    # Trả về kết quả dự đoán và các thông số đã nhập
    return render_template('index.html', 
                           prediction_text=f'Loài hoa Iris: {species}',
                           sepal_length=sepal_length,
                           sepal_width=sepal_width,
                           petal_length=petal_length,
                           petal_width=petal_width,
                           )

if __name__ == "__main__":
    app.run(debug=True)
