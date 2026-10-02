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
    # 2. SECCIÓN DE PRONÓSTICO PARA MAÑANA (SIN FILTRACIONES DE COLOR)
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
        # VISIÓN PRO (REVELADA)
        if prediccion == 2:
            col2.metric("Pronóstico (Día Siguiente)", "🟢 ALCISTA", "Mayor fuerza compradora", delta_color="normal")
        elif prediccion == 1:
            col2.metric("Pronóstico (Día Siguiente)", "🔴 BAJISTA", "Presión vendedora", delta_color="inverse")
        else:
            col2.metric("Pronóstico (Día Siguiente)", "🟡 LATERAL", "Sin tendencia clara", delta_color="off")

        col3.metric("Fuerza del Pronóstico", f"{max(prob_alcista, prob_bajista, prob_lateral):.1f}%", "Frente al 33% normal de azar", delta_color="off")

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
                f"**Consolidación / Rango:** La mayor probabilidad (**{prob_lateral:.1f}%**) indica mercado en pausa sin catalizador claro."
            )
    else:
        # VISIÓN PÚBLICA (SIN NINGUNA PISTA NI FLECHA DE COLOR)
        col2.metric("Pronóstico (Día Siguiente)", "🔒 BLOQUEADO")
        col3.metric("Fuerza del Pronóstico", "🔒 OCULTO")

        st.info("🔒 **Desglose de Probabilidades y Lectura Algorítmica Reservadas:** El cálculo predictivo de mañana y la lectura estratégica del modelo están protegidos para miembros activos.")
        st.markdown(
            "> 🛡️ **Ventaja Competitiva Protegida:** Para conocer la dirección probabilística de la siguiente sesión, adquiere una de las membresías al pie de la página o ingresa tu clave Pro en la barra lateral."
        )

    # -------------------------------------------------------------
    # 3. GRÁFICA DE EVOLUCIÓN HISTÓRICA (NÍTIDA Y PROFESIONAL)
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Evolución del Precio y Tendencias")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["Close"], mode="lines", name="Precio de Cierre", line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["SMA_40"], mode="lines", name="Tendencia Corto Plazo", line=dict(color="#f59e0b", dash="dot")))
    fig.add_trace(go.Scatter(x=datos_proc.index, y=datos_proc["SMA_160"], mode="lines", name="Tendencia Largo Plazo", line=dict(color="#7c3aed", dash="dash")))

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # 4. TARJETAS DE MEMBRESÍA AL ESTILO DEL EJEMPLO
    # -------------------------------------------------------------
    if not es_usuario_pro:
        st.markdown("---")
        st.subheader("💎 Desbloquea las Señales Diarias en Tiempo Real")
        st.write("Selecciona el plan que mejor se adapte a tu ritmo de trading para recibir tu clave de acceso inmediata:")

        # Inyección de estilos exactos a la tarjeta de referencia
        st.markdown(
            """
            <style>
            .pricing-wrapper {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 480px;
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
            }
            .pricing-wrapper-featured {
                background: #4f46e5;
                border: 1px solid #4338ca;
                border-radius: 14px;
                padding: 16px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 480px;
                color: #ffffff;
                box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.25);
            }
            .inner-box {
                background: #f8fafc;
                border-radius: 10px;
                padding: 24px 16px;
                text-align: center;
                margin-bottom: 16px;
            }
            .inner-box-featured {
                background: transparent;
                padding: 24px 16px;
                text-align: center;
                margin-bottom: 16px;
            }
            .plan-name {
                font-size: 1.15rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 8px;
            }
            .plan-name-white {
                font-size: 1.15rem;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 8px;
            }
            .plan-price-num {
                font-size: 2.5rem;
                font-weight: 800;
                color: #4f46e5;
                line-height: 1;
                margin-bottom: 10px;
            }
            .plan-price-num-white {
                font-size: 2.5rem;
                font-weight: 800;
                color: #ffffff;
                line-height: 1;
                margin-bottom: 10px;
            }
            .plan-subtitle {
                font-size: 0.85rem;
                color: #64748b;
                min-height: 38px;
                margin-bottom: 16px;
            }
            .plan-subtitle-white {
                font-size: 0.85rem;
                color: #e0e7ff;
                min-height: 38px;
                margin-bottom: 16px;
            }
            .btn-action-purple {
                display: block;
                width: 100%;
                background: #4f46e5;
                color: #ffffff !important;
                text-align: center;
                padding: 10px;
                border-radius: 8px;
                font-weight: 700;
                text-decoration: none;
                font-size: 0.95rem;
            }
            .btn-action-white {
                display: block;
                width: 100%;
                background: #ffffff;
                color: #4f46e5 !important;
                text-align: center;
                padding: 10px;
                border-radius: 8px;
                font-weight: 700;
                text-decoration: none;
                font-size: 0.95rem;
            }
            .feature-list {
                font-size: 0.88rem;
                color: #475569;
                line-height: 1.6;
                padding-left: 18px;
                margin-top: 8px;
            }
            .feature-list-white {
                font-size: 0.88rem;
                color: #f8fafc;
                line-height: 1.6;
                padding-left: 18px;
                margin-top: 8px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        col_t1, col_t2, col_t3 = st.columns(3)

        with col_t1:
            st.markdown(
                """
                <div class="pricing-wrapper">
                    <div class="inner-box">
                        <div class="plan-name">Pase Semanal</div>
                        <div class="plan-price-num">$50</div>
                        <div class="plan-subtitle">Acceso de prueba ideal para validar las señales algorítmicas en tu semana operativa.</div>
                        <a href="#" class="btn-action-purple">Comprar</a>
                    </div>
                    <div>
                        <ol class="feature-list">
                            <li>5 sesiones bursátiles completas.</li>
                            <li>Pronóstico KNN diario para la siguiente sesión.</li>
                            <li>Desglose probabilístico de las 9 tecnológicas.</li>
                        </ol>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_t2:
            st.markdown(
                """
                <div class="pricing-wrapper">
                    <div class="inner-box">
                        <div class="plan-name">Pase Quincenal</div>
                        <div class="plan-price-num">$75</div>
                        <div class="plan-subtitle">Para traders activos que buscan continuidad con un 25% de ahorro semanal.</div>
                        <a href="#" class="btn-action-purple">Comprar</a>
                    </div>
                    <div>
                        <ol class="feature-list">
                            <li>10 sesiones de mercado continuas.</li>
                            <li>Actualización diaria al cierre de Wall Street.</li>
                            <li>Consulta continua de métricas y tendencias.</li>
                        </ol>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_t3:
            st.markdown(
                """
                <div class="pricing-wrapper-featured">
                    <div class="inner-box-featured">
                        <div class="plan-name-white">⭐ Pase Mensual (Recomendado)</div>
                        <div class="plan-price-num-white">$125</div>
                        <div class="plan-subtitle-white">Máxima consistencia: solo $31.25 MXN por semana (37% de descuento).</div>
                        <a href="#" class="btn-action-white">Comprar</a>
                    </div>
                    <div>
                        <ol class="feature-list-white">
                            <li>Acceso total por 1 mes calendario completo (~22 sesiones).</li>
                            <li>Señales de las 9 megacaps tecnológicas sin límites.</li>
                            <li>Soporte de actualización y acceso prioritario a mejoras.</li>
                        </ol>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
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
