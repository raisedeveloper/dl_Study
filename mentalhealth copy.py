import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import pandas as pd
from tensorflow.keras.models import load_model



# In[7]:


# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format


# In[8]:


# 1. 데이터 로드 및 캐싱
@st.cache_data
def load_data():
    df = pd.read_csv('dataset/mental_health_wearable_data.csv', encoding='utf-8')
    df = df.rename(columns={'Mental_Health_Condition ': 'Mental_Health_Condition'})
    return df


df = load_data()
df.columns = df.columns.str.strip()
print(df.columns)


# In[9]:


df.head()


# In[10]:


# 결측치 확인
print("데이터 결측치 확인:")
print(df.isnull().sum())

# 결측치 처리 (평균값으로 대체)
df.fillna(df.mean(), inplace=True)

# In[11]:


# 불필요한 컬럼 제거 (원핫 인코딩 없이도 잘 작동하도록)
df = df.drop(['Physical_Activity_Steps', 'Mood_Rating'], axis=1)


# In[12]:


# 전처리 후 데이터 확인
print("전처리 후 데이터 형태:")
print("데이터:", df.shape)


# In[13]:


# 데이터 컬럼 확인
print("\n데이터 컬럼:", df.columns.tolist())


# In[14]:


# 스케일링 : 데이터 정규화 - (MinMaxScaler)
scaler = MinMaxScaler()
X = df.drop('Mental_Health_Condition', axis=1)
y = df['Mental_Health_Condition']


# In[15]:


# 피처 수 확인
n_features = X.shape[1]
print(f"피처 수: {n_features}")


# In[16]:


X = scaler.fit_transform(X)


# In[17]:


# 특성 이름 저장 (SHAP 시각화에 사용)
feature_names = df.drop('Mental_Health_Condition', axis=1).columns.tolist()


# In[18]:


# train, validation set 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Set (학습 데이터) : 모델을 학습시키는 데 사용
# Validation Set (검증 데이터) : 학습된 모델을 평가하고, 하이퍼파라미터 튜닝에 사용
# Test Set (테스트 데이터) : 최종 모델의 성능을 확인하는 데 사용 (실제 예측에 가까움)
# 모델이 과적합하지 않도록 검증 데이터(Validation Set)를 생성하는 과정
# Train Set에서 학습한 모델을 Validation Set에서 테스트해보면서 최적의 하이퍼파라미터를 찾고, 모델의 일반화 성능을 평가


# In[19]:


# 데이터 형태 확인
print(f"X_train 형태(reshape 전): {X_train.shape}")
print(f"X_val 형태(reshape 전): {X_val.shape}")


# In[20]:


# LSTM 입력을 위한 reshape (samples, time steps, features)
# LSTM 입력에 맞게 데이터 형태 변경
X_train = X_train.reshape(X_train.shape[0], 1, n_features)
X_val = X_val.reshape(X_val.shape[0], 1, n_features)


# In[21]:


# reshape 후 형태 확인
print(f"X_train 형태(reshape 후): {X_train.shape}")
print(f"X_val 형태(reshape 후): {X_val.shape}")


# In[22]:


# 2. LSTM 모델 구축(생성)
# model = Sequential()
# model.add(LSTM(64, input_shape=(1, n_features), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(32, return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))  # 정신건강이 좋을 가능성에 대한 이진 분류(1, 0)이므로 sigmoid 사용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2

model = Sequential()
model.add(LSTM(64, input_shape=(1, n_features), return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())  # 추가
model.add(Dropout(0.3))  # 기존 0.2 → 0.3으로 증가
model.add(GaussianNoise(0.1))  # 추가

model.add(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))  # 기존 0.2 → 0.3으로 증가
model.add(Dense(1, activation='sigmoid'))  # 이진 분류

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # binary_crossentropy : 이진 분류에 사용 (ex. 0, 1 등)


# In[23]:


model.summary()


# In[24]:


# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# In[25]:


# Early Stopping - verbose=1로 설정하여 진행상황 확인
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=0,          # 1번의 에폭 동안 개선이 없으면 학습 중단
    restore_best_weights=True,    #  가장 좋은 성능을 보였던 모델의 가중치(weights)를 복원
    verbose=1             # 중단 시 메시지 출력
)


# In[26]:


# 3. 모델 훈련 - verbose=1로 설정하여 진행상황 확인
epochs = 100
batch_size = 32

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1             # 훈련 진행상황 출력
)


# In[27]:


# 학습이 몇 번째 에폭에서 멈췄는지 확인
actual_epochs = len(history.history['loss'])
print(f"학습이 {actual_epochs}번째 에폭에서 완료되었습니다.")


# In[28]:


# 4. 예측 및 평가
# Validation set 평가
y_pred_val_proba = model.predict(X_val, verbose=0)

y_pred_val = (y_pred_val_proba > 0.5).astype(int)    # 0.5를 기준으로 분류, astype(int) : True는 1로, False는 0


# In[29]:


# ROC Curve 및 AUC 계산 (Validation set)
fpr, tpr, thresholds = roc_curve(y_val, y_pred_val_proba)
roc_auc = auc(fpr, tpr)


# In[30]:


# Confusion Matrix (Validation set)
cm = confusion_matrix(y_val, y_pred_val)
print("Confusion Matrix (Validation Set):\n", cm)


# In[31]:


# Classification Report (Validation set)
print("\nClassification Report (Validation Set):\n", classification_report(y_val, y_pred_val))


# In[32]:


# 5. 결과 시각화 및 통계 분석
# 학습 곡선(과정) 시각화 (loss, accuracy)

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
plt.tight_layout()
plt.show()


# In[33]:


# ROC Curve 시각화 (Validation set)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[34]:


# 실제값 vs 예측값 시각화 (Validation set - 처음 30개 샘플)
plt.figure(figsize=(10, 4))
plt.plot(y_val[:30].values, label='Actual', marker='o')
plt.plot(y_pred_val[:30].flatten(), label='Predicted', marker='x')  # flatten() 추가
plt.title('Actual vs Predicted (First 30 Samples - Validation Set)')
plt.xlabel('Sample Index')
plt.ylabel('Mental_Health_Condition')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)  # 그리드 추가
plt.show()


# In[35]:


# Confusion Matrix 시각화 (Validation set)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad_Mental_Health_Condition', 'Good_Mental_Health_Condition'], yticklabels=['Bad_Mental_Health_Condition', 'Good_Mental_Health_Condition'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Validation Set)')
plt.show()

# 정신건강 분포 확인
plt.figure(figsize=(10, 6))
sns.countplot(x='Mental_Health_Condition', data=df, palette='viridis')
plt.title('측정 대상자 분포', fontsize=15)
plt.xlabel('정신 건강 여부 (0: 나쁨, 1: 좋음)', fontsize=12)
plt.ylabel('인원 수', fontsize=12)


# 모델 저장 (선택 사항)
model.save('model/rnn_LSTM_pred_mentalHealth.h5')
print("모델이 'rnn_LSTM_pred_mentalHealth.h5' 파일로 저장되었습니다.")


# # 데이터 불러오기
# df = pd.read_csv('dataset/mental_health_wearable_data.csv', encoding='cp949')

# # 프로파일링 리포트 생성
# profile = ProfileReport(
#     df,
#     title="EDA 보고서",
#     explorative=True,
#     html={
#         'style': {
#             'theme': 'united'  # 허용된 theme 중 하나로 변경
#         }
#     }
# )

# # 리포트 저장 및 출력
# profile.to_file("./report/eda_report.html")

# # 보고서를 HTML 파일로 저장
# output_file = 'report/mental_health_profiling_report.html'
# profile.to_file(output_file)
# print(f"프로파일링 보고서가 생성되었습니다: {output_file}")