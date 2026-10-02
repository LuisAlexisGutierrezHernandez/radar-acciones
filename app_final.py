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

# Claves de acceso válidas (incluye tu clave permanente de fundador)
CLAVES_VALIDAS = ["FOUNDER_MASTER_2026", "RADAR2026", "PRO_MEMBER_50"]

# Lista de acciones disponibles exacta
tickers_default = ["NVDA", "MU", "AMD", "INTC", "AVGO", "GOOG", "META", "MSFT", "ORCL"]

# Barra lateral limpia
st.sidebar.header("⚙️ Configuración")
seleccion_ticker = st.sidebar.selectbox("Selecciona la acción a revisar:", tickers_default)
dias_analisis = st.sidebar.slider("Años de historia para analizar:", min_value=2, max_value=5, value=5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Acceso Suscriptor Pro")
clave_ingresada = st.sidebar.text_input("Ingresa tu Clave de Acceso:", type="password", placeholder="Ingresa tu clave...")

es_usuario_pro = clave_ingresada in CLAVES_VALIDAS

if es_usuario_pro:
    st.sidebar.success("✅ Acceso Pro Activo (En Vivo)")
elif clave_ingresada:
    st.sidebar.error("❌ Clave no válida")
else:
    st.sidebar.info("Modo libre activo. Desbloquea la señal de mañana al final.")

boton_analizar = st.sidebar.button("🔄 Actualizar Datos y Pronóstico")

st.title("📈 Radar Inteligente de Acciones Tecnológicas")
st.write(
    "Esta plataforma analiza el comportamiento reciente de las principales empresas "
    "tecnológicas y proyecta su tendencia más probable para la siguiente jornada con base en patrones históricos."
)

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

    future_return = d["Close"].shift(-3).pct_change(3)
    d["Target"] = 0
    d.loc[future_return > 0.01, "Target"] = 2
    d.loc[future_return < -0.01, "Target"] = 1

    return d.dropna().copy()

with st.spinner("Consultando datos reales de mercado..."):
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

    # -------------------------------------------------------------
    # 1. TABLA PÚBLICA: ÚLTIMAS 5 SESIONES REALES CERRADAS
    # -------------------------------------------------------------
    st.subheader(f"📋 Auditoría de los Últimos 5 Días Reales: {seleccion_ticker}")
    st.write("Verifica el comportamiento y proyecciones calculadas por el modelo en las 5 sesiones hábiles más recientes:")

    ultimas_5_X = X.iloc[-6:-1]
    preds_5 = modelo.predict(ultimas_5_X)
    probs_5 = modelo.predict_proba(ultimas_5_X)

    filas = []
    for idx, (fecha, row_x) in enumerate(ultimas_5_X.iterrows()):
        f_str = fecha.strftime("%Y-%m-%d")
        pr_cierre = datos_crudos.loc[fecha, "Close"]
        p_clase = preds_5[idx]
        p_probs = probs_5[idx]
        p_dict_temp = {clase: pr for clase, pr in zip(modelo.classes_, p_probs)}
        
        etiqueta = "🟢 Alcista" if p_clase == 2 else ("🔴 Bajista" if p_clase == 1 else "🟡 Lateral")
        fuerza = max(p_dict_temp.get(2, 0), p_dict_temp.get(1, 0), p_dict_temp.get(0, 0)) * 100
        
        filas.append({
            "Fecha Sesión Evaluada": f_str,
            "Precio Cierre": f"${pr_cierre:,.2f} USD",
            "Pronóstico KNN": etiqueta,
            "Fuerza Probabilística": f"{fuerza:.1f}%"
        })

    df_ultimos_5 = pd.DataFrame(filas)
    st.dataframe(df_ultimos_5, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 2. PRONÓSTICO PARA MAÑANA (CANDADO / TAPIZADO PARA NO SUSCRIPTORES)
    # -------------------------------------------------------------
    st.markdown("---")

    valores_hoy = X.iloc[[-1]]
    prediccion = modelo.predict(valores_hoy)[0]
    probabilidades = modelo.predict_proba(valores_hoy)[0]

    prob_dict = {clase: prob for clase, prob in zip(modelo.classes_, probabilidades)}
    prob_alcista = prob_dict.get(2, 0.0) * 100
    prob_bajista = prob_dict.get(1, 0.0) * 100
    prob_lateral = prob_dict.get(0, 0.0) * 100

    precio_actual = float(datos_crudos["Close"].iloc[-1])
    precio_anterior = float(datos_crudos["Close"].iloc[-2])
    cambio_hoy = ((precio_actual - precio_anterior) / precio_anterior) * 100
    fecha_hoy = datos_crudos.index[-1].strftime("%Y-%m-%d")

    st.subheader(f"📌 Resumen para la Siguiente Sesión: {seleccion_ticker}")
    st.caption(f"📅 **Datos base de cierre analizados:** {fecha_hoy} | **Proyección generada para:** Siguiente sesión de mercado")

    col1, col2, col3 = st.columns(3)
    col1.metric("Último Precio de Cierre", f"${precio_actual:,.2f} USD", f"{cambio_hoy:+.2f}% hoy")

    if es_usuario_pro:
        # VISIÓN SUSCRIPTOR PRO
        if prediccion == 2:
            col2.metric("Pronóstico (Día Siguiente)", "🟢 ALCISTA", "Mayor fuerza compradora")
        elif prediccion == 1:
            col2.metric("Pronóstico (Día Siguiente)", "🔴 BAJISTA", "Presión vendedora")
        else:
            col2.metric("Pronóstico (Día Siguiente)", "🟡 LATERAL", "Sin tendencia clara")

        col3.metric("Fuerza del Pronóstico", f"{max(prob_alcista, prob_bajista, prob_lateral):.1f}%", "Frente al 33% normal de azar")

        st.subheader("🎯 Desglose de Probabilidades")
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

        st.subheader("💡 Lectura clara para el inversionista")
        if prediccion == 2:
            st.success(
                f"**Predominio Comprador:** De los 50 momentos históricos más parecidos al cierre evaluado ({fecha_hoy}), "
                f"en el **{prob_alcista:.1f}%** de las ocasiones el precio subió. La balanza estadística favorece compras."
            )
        elif prediccion == 1:
            st.warning(
                f"**Presión Vendedora:** En el **{prob_bajista:.1f}%** de los escenarios similares la acción corrigió a la baja. "
                f"Solo un **{prob_alcista:.1f}%** logró subir. Conviene cautela."
            )
        else:
            st.info(
                f"**Consolidación / Rango:** La mayor probabilidad (**{prob_lateral:.1f}%**) indica mercado lateral sin catalizador claro."
            )
    else:
        # VISIÓN PÚBLICA (BLOQUEADA Y TAPADA CON CANDADO)
        col2.metric("Pronóstico (Día Siguiente)", "🔒 BLOQUEADO", "Exclusivo Suscriptores")
        col3.metric("Fuerza del Pronóstico", "🔒 OCULTO", "Requiere Clave Pro")

        st.info("🔒 **Desglose de Probabilidades y Lectura Algorítmica Reservadas:** El cálculo predictivo para la jornada de mañana está resguardado para miembros activos.")
        st.markdown(
            "> 🛡️ **Contenido Protegido:** Ingresa tu clave en la barra lateral o suscríbete a uno de los planes al pie de la página para desbloquear la proyección direccional y las probabilidades exactas."
        )

    # -------------------------------------------------------------
    # 3. GRÁFICA DE EVOLUCIÓN HISTÓRICA (SIEMPRE VISIBLE)
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Evolución del Precio y Tendencias")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["Close"], mode="lines", name="Precio de Cierre", line=dict(color="#0284c7", width=2)))
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["SMA_40"], mode="lines", name="Tendencia Corto Plazo", line=dict(color="#f59e0b", dash="dot")))
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["SMA_160"], mode="lines", name="Tendencia Largo Plazo", line=dict(color="#8b5cf6", dash="dash")))

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 4. TARJETAS DE MEMBRESÍA LIMPIAS Y FORMATEADAS
    # -------------------------------------------------------------
    if not es_usuario_pro:
        st.markdown("---")
        st.subheader("💎 Desbloquea las Señales en Tiempo Real para Mañana")
        st.write("Adquiere tu clave de acceso inmediata y opera con la ventaja cuantitativa del radar:")

        col_p1, col_p2, col_p3 = st.columns(3)

        with col_p1:
            with st.container(border=True):
                st.subheader("Pase Semanal")
                st.markdown("### $50 MXN")
                st.caption("5 sesiones hábiles de mercado")
                st.markdown(
                    """
                    * ✅ Señal algorítmica KNN para la siguiente sesión
                    * ✅ Desglose probabilístico de las 9 tecnológicas
                    * ✅ Acceso inmediato por 1 semana completa
                    * 🎯 Ideal para validar la herramienta
                    """
                )

        with col_p2:
            with st.container(border=True):
                st.subheader("Pase Quincenal")
                st.markdown("### $75 MXN")
                st.caption("10 sesiones hábiles de mercado")
                st.markdown(
                    """
                    * ✅ **Ahorro del 25%** frente al plan semanal
                    * ✅ Actualización diaria al cierre de Wall Street
                    * ✅ Acceso continuo por 2 semanas
                    * 🎯 Para swing traders activos
                    """
                )

        with col_p3:
            with st.container(border=True):
                st.markdown("**:blue[⭐ RECOMENDADO]**")
                st.subheader("Pase Mensual")
                st.markdown("### $125 MXN")
                st.caption("Acceso continuo por 1 mes calendario")
                st.markdown(
                    """
                    * ✅ **Costo equivalente a solo $31.25 MXN/semana**
                    * ✅ **37% de descuento total**
                    * ✅ Señales diarias para todo el radar (NVDA, AMD, etc.)
                    * ✅ Máxima consistencia operativa
                    """
                )

    # -------------------------------------------------------------
    # 5. DESLINDE LEGAL
    # -------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "⚠️ **Aviso Legal y de Responsabilidad:** Esta plataforma es una herramienta tecnológica de análisis estadístico "
        "y modelado cuantitativo con fines informativos y educativos. La información aquí presentada **no constituye bajo ninguna circunstancia "
        "una recomendación personalizada de inversión, asesoría patrimonial ni una invitación para comprar o vender activos bursátiles**. "
        "El rendimiento histórico no garantiza resultados futuros. Cada usuario es plenamente responsable de sus decisiones financieras, "
        "de la gestión de su capital y de determinar el momento oportuno de entrada o salida en el mercado."
    )
