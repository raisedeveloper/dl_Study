import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/mental_health_wearable_data.csv', encoding='utf-8')
    df = df.rename(columns={'Mental_Health_Condition ': 'Mental_Health_Condition'})
    return df

df = load_data()
df.columns = df.columns.str.strip()

# 불필요한 컬럼 제거
df = df.drop(['Physical_Activity_Steps', 'Mood_Rating'], axis=1)

# 스케일링
scaler = MinMaxScaler()
X = df.drop('Mental_Health_Condition', axis=1)
y = df['Mental_Health_Condition']
X = scaler.fit_transform(X)

# 데이터 분할 (Train 70%, Validation 30%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# LSTM 입력 형태 변환
n_features = X.shape[1]
X_train = X_train.reshape(X_train.shape[0], 1, n_features)
X_val = X_val.reshape(X_val.shape[0], 1, n_features)

# 2. LSTM 모델 구축
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)), input_shape=(1, n_features)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류

# 옵티마이저 & 모델 컴파일
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# EarlyStopping 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# 3. 모델 훈련
history = model.fit(
    X_train, y_train,
    epochs=150,  # 증가하여 충분히 학습
    batch_size=64,  # 배치 크기 조정
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# 4. 모델 평가 및 예측
y_pred_val_proba = model.predict(X_val, verbose=0)
y_pred_val = (y_pred_val_proba > 0.5).astype(int)

# 성능 평가
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_val))
print("\nClassification Report:\n", classification_report(y_val, y_pred_val))

# 5. 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 모델 저장
model.save('model/optimized_LSTM_mentalHealth.h5')
