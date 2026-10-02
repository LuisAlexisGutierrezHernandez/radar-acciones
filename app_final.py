import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import pytz
import json
import os
import secrets

# ==========================================
# CONFIGURACIÓN DE PÁGINA (ESTILO FINVIZ DARK)
# ==========================================
st.set_page_config(
    page_title="MarketVision AI - Finviz Edition",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados inspirados en Finviz Dark Mode
st.markdown("""
<style>
    /* Fondo general estilo Finviz */
    .stApp {
        background-color: #12151e;
        color: #e4e7eb;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Barra lateral */
    section[data-testid="stSidebar"] {
        background-color: #1a1e2c;
        border-right: 1px solid #2a3142;
    }
    
    /* Tarjetas y contenedores */
    .metric-card {
        background: #1b2030;
        border: 1px solid #2d3648;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    /* Botones estilo Finviz */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    
    /* Precios y planes */
    .pricing-badge {
        background-color: #20273a;
        border-left: 4px solid #10b981;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    
    /* Disclaimer educativo */
    .disclaimer-box {
        background-color: #1e2433;
        border: 1px solid #4b5563;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)

TICKERS = ["NVDA", "MU", "AMD", "INTC", "AVGO", "GOOG", "META", "MSFT", "ORCL"]
ADMIN_KEY = "ADMIN-MASTER-PERMANENT-2026"
LICENSES_FILE = "licencias.json"
CDMX_TZ = pytz.timezone("America/Mexico_City")

# ==========================================
# GESTIÓN Y REGLAS DE LICENCIAS (DÍAS HÁBILES)
# ==========================================
def init_licenses_file():
    """Genera las 100 claves por plan si no existen previamente."""
    if not os.path.exists(LICENSES_FILE):
        data = {"keys": {}}
        plans = {
            "5_DIAS": ("SEM", 5),
            "10_DIAS": ("QUIN", 10),
            "21_DIAS": ("MES", 21)
        }
        for plan_name, (prefix, days) in plans.items():
            for _ in range(100):
                code = f"{prefix}-{secrets.token_hex(4).upper()}"
                data["keys"][code] = {
                    "plan": plan_name,
                    "dias_habiles": days,
                    "active": False,
                    "email": None,
                    "fecha_activacion": None,
                    "fecha_vencimiento": None
                }
        with open(LICENSES_FILE, "w") as f:
            json.dump(data, f, indent=2)

def load_licenses():
    init_licenses_file()
    with open(LICENSES_FILE, "r") as f:
        return json.load(f)

def save_licenses(data):
    with open(LICENSES_FILE, "w") as f:
        json.dump(data, f, indent=2)

def calculate_expiration_date(start_dt, business_days_needed):
    """
    Calcula fecha de vencimiento según reglas:
    - Corte a las 2:00 PM CDMX. Si se envía después de las 2:00 PM o en fin de semana,
      empieza a contar a partir del siguiente día hábil (lunes si es fin de semana).
    """
    current = start_dt
    # Si pasa de las 2:00 PM o es fin de semana (sábado=5, domingo=6)
    if current.time() >= time(14, 0) or current.weekday() >= 5:
        current = current + timedelta(days=1)
        current = datetime.combine(current.date(), time(9, 0))
        current = CDMX_TZ.localize(current)

    # Asegurarse de que el primer día de conteo sea hábil
    while current.weekday() >= 5:
        current += timedelta(days=1)

    counted_days = 0
    test_day = current
    while counted_days < business_days_needed:
        if test_day.weekday() < 5:  # Lunes a Viernes
            counted_days += 1
            if counted_days == business_days_needed:
                break
        test_day += timedelta(days=1)
    
    # Vence al cierre de ese día hábil (23:59:59)
    expiration = datetime.combine(test_day.date(), time(23, 59, 59))
    return CDMX_TZ.localize(expiration)

def validate_user_access(email, key):
    email = email.strip().lower()
    key = key.strip()
    
    # 1. Acceso de Administrador Permanente
    if key == ADMIN_KEY:
        return True, "Acceso Maestro Permanente Activo", "Ilimitado"

    licenses = load_licenses()["keys"]
    if key not in licenses:
        return False, "La clave de acceso ingresada no existe.", None

    lic = licenses[key]
    now_cdmx = datetime.now(CDMX_TZ)

    # 2. Si la clave aún no está activada, se activa ligada al correo
    if not lic["active"]:
        lic["active"] = True
        lic["email"] = email
        lic["fecha_activacion"] = now_cdmx.strftime("%Y-%m-%d %H:%M:%S")
        exp_dt = calculate_expiration_date(now_cdmx, lic["dias_habiles"])
        lic["fecha_vencimiento"] = exp_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        full_data = load_licenses()
        full_data["keys"][key] = lic
        save_licenses(full_data)
        
        return True, f"Clave activada con éxito para {email}.", lic["fecha_vencimiento"]

    # 3. Si ya está activa, verificar que pertenezca al mismo correo
    if lic["email"] != email:
        return False, "Esta clave ya fue activada con otra cuenta de correo.", None

    # 4. Verificar vigencia
    exp_dt = CDMX_TZ.localize(datetime.strptime(lic["fecha_vencimiento"], "%Y-%m-%d %H:%M:%S"))
    if now_cdmx > exp_dt:
        return False, f"Tu suscripción venció el {lic['fecha_vencimiento']} (CDMX).", None

    return True, "Suscripción activa.", lic["fecha_vencimiento"]

# ==========================================
# MOTOR DE MACHINE LEARNING & DATOS
# ==========================================
@st.cache_data(ttl=3600)
def fetch_and_clean_data(tickers_list):
    raw_data = yf.download(tickers_list, period="5y", interval="1d", group_by="ticker", progress=False)
    cleaned = {}
    for t in tickers_list:
        df = raw_data[t].dropna().copy()
        df = df[(df["Volume"] > 0) & (df["High"] != df["Low"])].copy()
        
        # Medias y Pendientes
        df["SMA_40"] = df["Close"].rolling(window=40).mean()
        df["SMA_80"] = df["Close"].rolling(window=80).mean()
        df["SMA_160"] = df["Close"].rolling(window=160).mean()
        df["Slope_SMA_40"] = df["SMA_40"].pct_change(5)
        df["Slope_SMA_80"] = df["SMA_80"].pct_change(5)
        df["Slope_SMA_160"] = df["SMA_160"].pct_change(5)

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Rango
        df["Relative_Range"] = (df["High"] - df["Low"]) / df["Close"]

        # Variación diaria reciente (para mapa de calor Finviz)
        df["Pct_Change_Today"] = df["Close"].pct_change() * 100

        # Target a 3 días
        future_return = df["Close"].shift(-3).pct_change(3)
        df["Target"] = 0
        df.loc[future_return > 0.01, "Target"] = 2
        df.loc[future_return < -0.01, "Target"] = 1

        cleaned[t] = df.dropna().copy()
    return cleaned

def compute_market_forecast(processed_data):
    features = ["Slope_SMA_40", "Slope_SMA_80", "Slope_SMA_160", "RSI", "Relative_Range"]
    label_friendly = {
        2: "Subida Estimada (+1% o más)",
        1: "Caída Estimada (-1% o más)",
        0: "Movimiento Neutral / Sin gran cambio"
    }
    
    rows = []
    for t in TICKERS:
        df = processed_data[t]
        X_all = df[features]
        y_all = df["Target"]

        model = KNeighborsClassifier(n_neighbors=50)
        model.fit(X_all, y_all)

        last_row = X_all.iloc[[-1]]
        pred = model.predict(last_row)[0]
        prob = model.predict_proba(last_row)[0].max()
        
        last_price = df["Close"].iloc[-1]
        pct_today = df["Pct_Change_Today"].iloc[-1]

        rows.append({
            "Acción": t,
            "Último Precio": f"${last_price:.2f}",
            "Rendimiento Diario": pct_today,
            "Pronóstico para la Próxima Sesión": label_friendly[pred],
            "Nivel de Seguridad del Modelo": f"{prob*100:.1f}%",
            "Codigo_Senal": pred
        })
    return pd.DataFrame(rows)

# ==========================================
# BARRA LATERAL: LOGIN Y SUSCRIPCIONES
# ==========================================
init_licenses_file()

with st.sidebar:
    st.image("https://finviz.com/favicon.ico", width=28)
    st.title("Acceso & Cuenta")
    
    user_email = st.text_input("Correo Electrónico:", placeholder="tunombre@ejemplo.com")
    user_key = st.text_input("Clave de Licencia:", type="password", placeholder="Ingresa tu clave")
    
    is_authenticated = False
    status_msg = ""
    expiration_date = None

    if st.button("Iniciar Sesión"):
        if not user_email or not user_key:
            st.error("Por favor ingresa tu correo y tu clave.")
        else:
            is_authenticated, status_msg, expiration_date = validate_user_access(user_email, user_key)
            if is_authenticated:
                st.session_state["auth"] = True
                st.session_state["email"] = user_email
                st.session_state["exp"] = expiration_date
                st.success("Sesión iniciada con éxito")
            else:
                st.error(status_msg)
                
    if st.session_state.get("auth", False):
        is_authenticated = True
        st.markdown(f"**Usuario:** `{st.session_state.get('email')}`")
        st.markdown(f"**Vencimiento:** `{st.session_state.get('exp')}`")
        if st.button("Cerrar Sesión"):
            st.session_state.clear()
            st.rerun()

    st.markdown("---")
    st.subheader("💳 Planes de Suscripción")
    st.markdown("""
    <div class="pricing-badge">
        <b>Plan 1 Semana (5 días bursátiles):</b> $50 MXN
    </div>
    <div class="pricing-badge">
        <b>Plan 2 Semanas (10 días bursátiles):</b> $75 MXN
    </div>
    <div class="pricing-badge">
        <b>Plan 1 Mes (~21 días bursátiles):</b> $125 MXN
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📲 ¿Cómo contratar?")
    st.markdown("""
    1. Envía un mensaje por WhatsApp al:
       **[+52 56 5019 7036](https://wa.me/525650197036)**
    2. Solicita la CLABE interbancaria para transferir según tu plan preferido.
    3. Al confirmar tu pago recibirás tu clave única e intransferible.
    """)

# ==========================================
# PANEL PRINCIPAL
# ==========================================
st.markdown("## 📊 Monitor de Mercado & Mapa de Calor (Estilo Finviz)")
st.caption("Visualización clara del comportamiento reciente de las principales empresas tecnológicas y predicciones cuantitativas inteligentes.")

with st.spinner("Sincronizando datos de mercado en tiempo real..."):
    data_dict = fetch_and_clean_data(TICKERS)
    forecast_df = compute_market_forecast(data_dict)

# ------------------------------------------
# MAPA DE CALOR ESTILO FINVIZ (ROJO A VERDE INTENSO)
# ------------------------------------------
st.subheader("🗺️ Mapa de Rendimiento Diario")
st.markdown("Representa el desempeño de hoy: verde para ganancias y rojo para caídas, tal como en Finviz[cite: 2, 3].")

# Ajuste de escala idéntico a Finviz: -3% o menos (rojo intenso) a +3% o más (verde intenso)
fig_map = px.treemap(
    forecast_df,
    path=[px.Constant("Portafolio Tecnológico"), "Acción"],
    values=[1] * len(forecast_df),  # Bloques de tamaño uniforme y limpio
    color="Rendimiento Diario",
    color_continuous_scale=[
        [0.0, "#f63538"],    # Rojo intenso Finviz (-3% o inferior)[cite: 3]
        [0.35, "#8b2c34"],   # Rojo intermedio (-1%)[cite: 3]
        [0.5, "#303649"],    # Neutro / Gris azulado (0%)[cite: 3]
        [0.65, "#266b44"],   # Verde oscuro (+1%)[cite: 3]
        [1.0, "#00c853"]     # Verde intenso Finviz (+3% o mayor)[cite: 3]
    ],
    range_color=[-3.0, 3.0],
    custom_data=["Último Precio", "Rendimiento Diario"]
)

fig_map.update_traces(
    texttemplate="<b>%{label}</b><br>%{customdata[1]:+.2f}%<br>%{customdata[0]}",
    textposition="middle center",
    textfont=dict(size=18, color="white", family="Arial Black")
)

fig_map.update_layout(
    margin=dict(t=10, l=10, r=10, b=10),
    template="plotly_dark",
    paper_bgcolor="#12151e",
    plot_bgcolor="#12151e",
    height=420,
    coloraxis_colorbar=dict(
        title="Cambio (%)",
        ticks="outside",
        tickvals=[-3, -2, -1, 0, 1, 2, 3],
        ticktext=["-3%", "-2%", "-1%", "0%", "+1%", "+2%", "+3%"]
    )
)

st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------------------
# SECCIÓN DE PRONÓSTICOS PARA EL DÍA SIGUIENTE
# ------------------------------------------
st.markdown("---")
st.subheader("🎯 Pronósticos de Inteligencia Artificial para la Siguiente Sesión")

# Fecha de pronóstico explicada con claridad
cdmx_now = datetime.now(CDMX_TZ)
dia_semana = cdmx_now.weekday()
if dia_semana == 4:  # Viernes
    prox_sesion = "Lunes (Próxima sesión bursátil)"
elif dia_semana in [5, 6]: # Sábado o Domingo
    prox_sesion = "Lunes (Próxima sesión bursátil)"
else:
    prox_sesion = "Mañana"

st.info(f"📅 **Sesión analizada:** {cdmx_now.strftime('%d/%m/%Y')} | **Los pronósticos mostrados aplican para:** {prox_sesion}")

if not is_authenticated:
    st.warning("🔒 **Contenido Exclusivo para Suscriptores:** Los pronósticos predictivos calculados por el algoritmo están bloqueados. Adquiere tu suscripción o ingresa con tu clave para visualizarlos.")
    
    # Vista previa difuminada/demostrativa
    preview_df = forecast_df[["Acción", "Último Precio"]].copy()
    preview_df["Pronóstico para la Próxima Sesión"] = "🔒 Solo con suscripción activa"
    preview_df["Nivel de Seguridad del Modelo"] = "🔒 Bloqueado"
    st.dataframe(preview_df, use_container_width=True, hide_index=True)
else:
    st.success("✅ Acceso concedido a los pronósticos algorítmicos.")
    
    # Estilizado visual de las señales para facilitar la lectura sin tecnicismos
    display_df = forecast_df[["Acción", "Último Precio", "Rendimiento Diario", "Pronóstico para la Próxima Sesión", "Nivel de Seguridad del Modelo"]]
    
    st.dataframe(
        display_df.style.format({
            "Rendimiento Diario": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    with st.expander("ℹ️ ¿Cómo interpretar estos pronósticos de forma sencilla?"):
        st.markdown("""
        * **Subida Estimada:** La inteligencia artificial detecta patrones estadísticos similares a días donde la acción subió al menos un **1%** en los días posteriores.
        * **Caída Estimada:** El algoritmo detecta probabilidades altas de retroceso o corrección mayor al **1%**.
        * **Movimiento Neutral:** No se observa una tendencia clara; es probable que el precio se mantenga estable o en rango.
        * **Nivel de Seguridad:** Porcentaje de coincidencia entre los patrones históricos pasados y la situación técnica de hoy.
        """)

# ==========================================
# DESCARGO DE RESPONSABILIDAD LEGAL / EDUCATIVO
# ==========================================
st.markdown("""
<div class="disclaimer-box">
    <b>Aviso Legal Importante:</b> Esta herramienta digital ha sido creada con fines estrictamente educativos, informativos y de análisis computacional. 
    Bajo ninguna circunstancia representa una recomendación u orden de inversión personalizada, ni pretende sustituir la asesoría financiera calificada. Toda inversión en el mercado bursátil conlleva riesgos de pérdida de capital; cada usuario es plenamente responsable de sus decisiones patrimoniales.
</div>
""", unsafe_allow_html=True)
