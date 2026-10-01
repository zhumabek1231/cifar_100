import streamlit as st
import requests
from streamlit.runtime.uploaded_file_manager import UploadFileUrlInfo
api_url = 'http://127.0.0.1:8000/predict'
st.title('Предсказания Фото')

uploaded_file = st.file_uploader("Загрузи изображение", type=["png", "jpg", "jpeg"])


if st.button('Проверка'):
    if uploaded_file is not None:
        try:
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(api_url, files=files, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success(f'Результат: {result.get("Answer")}')
                st.image(uploaded_file, caption="Загруженное изображение")
            else:
                st.error(f'Ошибка: {response.status_code}')

        except requests.exceptions.RequestException:
            st.error('Не удалось подключиться к API')

    else:
        st.warnin  ("Сначала загрузи файл!")