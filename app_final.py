import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

# Configuración visual de la página
st.set_page_config(
    page_title="Radar de Acciones - Inteligencia de Mercado",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Radar Inteligente de Acciones Tecnológicas")
st.write(
    "Esta plataforma analiza el comportamiento reciente de las principales empresas "
    "tecnológicas y proyecta su tendencia más probable para la siguiente jornada con base en patrones históricos."
)

# Lista de acciones disponibles
tickers_default = ["NVDA", "MU", "AMD", "INTC", "AVGO", "GOOG", "META", "MSFT", "ORCL"]

# Barra lateral
st.sidebar.header("⚙️ Configuración")
seleccion_ticker = st.sidebar.selectbox("Selecciona la acción a revisar:", tickers_default)
dias_analisis = st.sidebar.slider("Años de historia para analizar:", min_value=2, max_value=5, value=5)

boton_analizar = st.sidebar.button("🔄 Actualizar Datos y Pronóstico")

@st.cache_data(ttl=3600)
def descargar_datos(ticker, period_years):
    df = yf.download(ticker, period=f"{period_years}y", interval="1d", auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna().copy()
    df = df[(df["Volume"] > 0) & (df["High"] != df["Low"])].copy()
    return df

def calcular_indicadores(df):
    d = df.copy()
    d["SMA_40"] = d["Close"].rolling(window=40).mean()
    d["SMA_80"] = d["Close"].rolling(window=80).mean()
    d["SMA_160"] = d["Close"].rolling(window=160).mean()

    d["Slope_SMA_40"] = d["SMA_40"].pct_change(5)
    d["Slope_SMA_80"] = d["SMA_80"].pct_change(5)
    d["Slope_SMA_160"] = d["SMA_160"].pct_change(5)

    delta = d["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI"] = 100 - (100 / (1 + rs))

    d["Relative_Range"] = (d["High"] - d["Low"]) / d["Close"]

    # 2: Sube (>1%), 1: Baja (<-1%), 0: Lateral / Rango
    future_return = d["Close"].shift(-3).pct_change(3)
    d["Target"] = 0
    d.loc[future_return > 0.01, "Target"] = 2
    d.loc[future_return < -0.01, "Target"] = 1

    return d.dropna().copy()

with st.spinner("Consultando datos en tiempo real de Wall Street..."):
    datos_crudos = descargar_datos(seleccion_ticker, dias_analisis)

if len(datos_crudos) < 200:
    st.error("No hay suficientes datos disponibles para esta empresa.")
else:
    datos_proc = calcular_indicadores(datos_crudos)
    features = ["Slope_SMA_40", "Slope_SMA_80", "Slope_SMA_160", "RSI", "Relative_Range"]

    X = datos_proc[features]
    y = datos_proc["Target"]

    modelo = KNeighborsClassifier(n_neighbors=50)
    modelo.fit(X, y)

    valores_hoy = X.iloc[[-1]]
    prediccion = modelo.predict(valores_hoy)[0]
    probabilidades = modelo.predict_proba(valores_hoy)[0]
    
    # Mapeo de probabilidades (0=Lateral, 1=Bajista, 2=Alcista)
    prob_dict = {clase: prob for clase, prob in zip(modelo.classes_, probabilidades)}
    prob_alcista = prob_dict.get(2, 0.0) * 100
    prob_bajista = prob_dict.get(1, 0.0) * 100
    prob_lateral = prob_dict.get(0, 0.0) * 100

    precio_actual = float(datos_crudos["Close"].iloc[-1])
    precio_anterior = float(datos_crudos["Close"].iloc[-2])
    cambio_hoy = ((precio_actual - precio_anterior) / precio_anterior) * 100

    # Tarjetas visuales de resumen
    st.subheader(f"📌 Resumen para la Siguiente Sesión: {seleccion_ticker}")
    col1, col2, col3 = st.columns(3)

    col1.metric("Último Precio", f"${precio_actual:,.2f} USD", f"{cambio_hoy:+.2f}% hoy")

    if prediccion == 2:
        col2.metric("Pronóstico (Día Siguiente)", "🟢 ALCISTA", "Mayor fuerza compradora")
    elif prediccion == 1:
        col2.metric("Pronóstico (Día Siguiente)", "🔴 BAJISTA", "Presión vendedora")
    else:
        col2.metric("Pronóstico (Día Siguiente)", "🟡 LATERAL", "Sin tendencia clara")

    col3.metric("Fuerza del Pronóstico", f"{max(prob_alcista, prob_bajista, prob_lateral):.1f}%", "Frente al 33% normal de azar")

    # Sección explicativa de probabilidades
    st.markdown("---")
    st.subheader("🎯 Desglose de Probabilidades (¿Por qué este pronóstico?)")
    st.write(
        "Al evaluar una acción existen **3 caminos posibles**: subir con fuerza, bajar con fuerza o mantenerse en calma. "
        "En un escenario puramente al azar, cada camino tendría solo un **33.3%**. Aquí te mostramos cómo se distribuye el 100% de los patrones históricos:"
    )

    c_up, c_flat, c_down = st.columns(3)
    with c_up:
        st.write(f"🟢 **Probabilidad Alcista: {prob_alcista:.1f}%**")
        st.progress(int(prob_alcista))
    with c_flat:
        st.write(f"🟡 **Probabilidad Lateral / Rango: {prob_lateral:.1f}%**")
        st.progress(int(prob_lateral))
    with c_down:
        st.write(f"🔴 **Probabilidad Bajista: {prob_bajista:.1f}%**")
        st.progress(int(prob_bajista))

    # Gráfica interactiva de precios
    st.markdown("---")
    st.subheader("📊 Evolución del Precio y Tendencia")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["Close"], mode="lines", name="Precio de Cierre", line=dict(color="#00D4B2", width=2)))
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["SMA_40"], mode="lines", name="Tendencia Corto Plazo", line=dict(color="#FFA500", dash="dot")))
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["SMA_160"], mode="lines", name="Tendencia Largo Plazo", line=dict(color="#A020F0", dash="dash")))

    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explicación clara al usuario
    st.markdown("---")
    st.subheader("💡 Lectura clara para el inversionista")
    if prediccion == 2:
        st.success(
            f"**Predominio Comprador:** De los 50 momentos históricos más parecidos a la situación de hoy, "
            f"en el **{prob_alcista:.1f}%** de las ocasiones el precio subió. El restante no es caída directa: solo un "
            f"**{prob_bajista:.1f}%** cayó y un **{prob_lateral:.1f}%** se mantuvo neutral. La balanza favorece compras."
        )
    elif prediccion == 1:
        st.warning(
            f"**Precaución:** En el **{prob_bajista:.1f}%** de los escenarios similares la acción corrigió a la baja. "
            f"Solo un **{prob_alcista:.1f}%** logró subir y un **{prob_lateral:.1f}%** se mantuvo lateral."
        )
    else:
        st.info(
            f"**Mercado en Espera:** La mayor probabilidad (**{prob_lateral:.1f}%**) indica consolidación. "
            f"No hay una ventaja clara entre compradores ({prob_alcista:.1f}%) y vendedores ({prob_bajista:.1f}%)."
        )

    # Deslinde de responsabilidad legal y ética
    st.markdown("---")
    st.caption(
        "⚠️ **Aviso Legal y de Responsabilidad:** Esta plataforma es una herramienta tecnológica de análisis estadístico "
        "y modelado cuantitativo con fines informativos y educativos. La información aquí presentada **no constituye bajo ninguna circunstancia "
        "una recomendación personalizada de inversión, asesoría patrimonial ni una invitación para comprar o vender activos bursátiles**. "
        "El rendimiento histórico no garantiza resultados futuros. Cada usuario es plenamente responsable de sus decisiones financieras, "
        "de la gestión de su capital y de determinar el momento oportuno de entrada o salida en el mercado."
    )