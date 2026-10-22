from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
st.set_page_config(page_title="Диана — STEM жылыжайы", page_icon="🌱")
st.title("🌱 STEM жылыжайы: суару болжамы")
st.caption("Диана • 200 жасанды жазба • Google Colab → Streamlit")
st.info("Оқу моделі шартты ережелерден жасалған деректермен оқытылған. Нақты агрономиялық ұсыныс емес.")
@st.cache_resource
def load_bundle():
    return joblib.load(Path(__file__).parent / "greenhouse_model.joblib")
try:
    bundle=load_bundle()
except FileNotFoundError:
    st.error("Алдымен Colab ноутбугін орындап, greenhouse_model.joblib файлын app.py жанына салыңыз.")
    st.stop()
with st.form("prediction"):
    t=st.number_input("Температура (°C)",15.0,35.0,30.0,0.1)
    a=st.number_input("Ауа ылғалдылығы (%)",30.0,90.0,40.0,0.1)
    s=st.number_input("Топырақ ылғалдылығы (%)",10.0,80.0,20.0,0.1)
    submit=st.form_submit_button("Болжам жасау")
if submit:
    row=pd.DataFrame([[t,a,s]],columns=bundle["features"])
    result=int(bundle["model"].predict(row)[0])
    if result: st.warning("💧 Суару қажет / Нужно поливать")
    else: st.success("✅ Суару қажет емес / Поливать не нужно")
    st.caption("Сандар сақталған StandardScaler арқылы түрлендіріліп, оқытылған модельге беріледі.")
    with st.expander("Нормаланған мәндер"):
        st.dataframe(pd.DataFrame(bundle["model"].named_steps["scaler"].transform(row),columns=bundle["features"]))
st.caption("Модель бағасы: тест жиынындағы accuracy = "+str(round(bundle["metrics"]["accuracy"],3)))
