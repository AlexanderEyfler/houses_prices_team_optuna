import streamlit as st
import joblib
import pandas as pd
import sklearn
import numpy as np
import category_encoders

from catboost import CatBoostRegressor
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from custom_process import MyTransformer, MissingValueReplacer


# Устанавливаем глобальную настройку для sklearn
set_config(transform_output="pandas")

# Загружаем конвейер предварительной обработки
preprocessor = joblib.load('preprocessing_pipeline.pkl')

# Загружаем модель CatBoost
model = CatBoostRegressor()
model.load_model('catboost_model_lam_data.cbm')

# Заголовок приложения
st.title('Прогнозирование цен на дома для Kaggle House Prices - Advanced Regression Techniques')

# Загрузка данных
uploaded_file = st.file_uploader("Загрузите ваш CSV-файл", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Данные загружены:')
    st.write(data.head())

    # Кнопка для запуска предсказания
    if st.button('Сделать предсказание'):
        # Предобработка данных
        data_preprocessed = preprocessor.transform(data)

        # Предсказание
        log_predictions = model.predict(data_preprocessed)
        predictions = np.exp(log_predictions)

        # Отображение предсказаний
        st.write('Предсказания:')
        st.write(predictions)
