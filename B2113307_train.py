import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

iris = pd.read_csv('iris.csv')

X = iris.drop(columns=['variety']) 
y = iris['variety']

# Mã hóa nhãn thành số
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Huấn luyện mô hình KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

# Lưu mô hình, scaler và label_encoder
joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("Đã lưu thành công mô hình, scaler và label_encoder.")
