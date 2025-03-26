import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

# 모델 로드 로직 개선
@st.cache_resource  # Streamlit의 캐싱 데코레이터 사용
def load_model():
    try:
        if os.path.exists('model/rnn_LSTM_pred_mentalHealth.h5'):
            return tf.keras.models.load_model('model/rnn_LSTM_pred_mentalHealth.h5') 
        else:
            st.error("⚠️ 모델 파일을 찾을 수 없습니다. 먼저 모델을 훈련해주세요.")
            return None
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

# 모델 로드
model = load_model()

# MinMaxScaler 직접 정의
scaler = MinMaxScaler()

# 스케일러 설정 (학습 시 사용했던 데이터 범위로 설정)
scaler.min_ = np.array([60, 4])  # 심박수, 수면시간의 최소값
scaler.scale_ = np.array([120, 9])  # 심박수, 수면시간의 최대값 기준 스케일

# 웹 앱 제목
st.title("🧘‍♀️정신 건강 예측 앱🧘")
st.write("💪사용자의 건강 데이터를 바탕으로 정신 건강 상태를 예측합니다.")

# 사용자 입력 받기
heart_rate = st.number_input("🫀평균 심박수 (bpm)", min_value=60, max_value=120, value=90)
sleep_hours = st.number_input("💤하루 평균 수면 시간 (시간)", min_value=4.0, max_value=9.0, value=4.5)

# 예측 버튼
if st.button("✔️예측하기"):
    if model is not None:
        try:
            # 입력 데이터를 배열로 변환
            user_input = np.array([[heart_rate, sleep_hours]])
            user_input_scaled = scaler.transform(user_input)  # 입력값 정규화
            user_input_reshaped = user_input_scaled.reshape(1, 1, -1)  # LSTM 모델 입력 형태로 변환

            prediction = model.predict(user_input_reshaped)
            prediction_prob = prediction[0][0]
            prediction_label = "정신 건강 양호" if prediction_prob > 0.5 else "정신 건강 나쁨"
            
            st.subheader("☑️예측 결과")
            st.write(f"모델의 예측 결과: **{prediction_label}**")
            st.write(f"예측 확률: **{prediction_prob:.2%}**")
            
            # 시각적 결과 추가
            if prediction_prob > 0.:
                st.success("👍좋은 정신 건강 상태입니다! 지속적으로 건강을 유지하세요!")
            else:
                st.warning("😣정신 건강 관리가 필요합니다. 충분한 휴식과 긍정적인 생활습관을 유지하세요.")
        
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")
    else:
        st.error("❌ 모델이 로드되지 않았습니다. 모델 훈련을 먼저 진행해주세요.")