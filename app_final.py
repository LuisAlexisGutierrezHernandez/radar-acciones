import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="AI Stock Predictor | Predicción Bursátil",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS PERSONALIZADO (estética profesional)
# =========================================================
st.markdown("""
<style>
    /* Fondo y tipografía */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main {
        background: linear-gradient(180deg, #0E1117 0%, #131A26 100%);
    }

    /* Header principal */
    .hero {
        background: linear-gradient(135deg, #1A1F2E 0%, #0E1117 100%);
        border: 1px solid #2A3142;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .hero h1 {
        color: #FAFAFA;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin: 0;
    }
    .hero .accent { color: #00D4AA; }

    /* Tarjetas KPI */
    .kpi-card {
        background: #1A1F2E;
        border: 1px solid #2A3142;
        border-radius: 14px;
        padding: 20px 22px;
        text-align: left;
        height: 100%;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: #00D4AA; }
    .kpi-label {
        color: #9CA3AF;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #FAFAFA;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .kpi-sub { color: #6B7280; font-size: 0.8rem; margin-top: 4px; }

    /* Badges de señal */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-bull { background: rgba(0, 212, 170, 0.15); color: #00D4AA; border: 1px solid #00D4AA; }
    .badge-bear { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid #EF4444; }
    .badge-flat { background: rgba(234, 179, 8, 0.15); color: #EAB308; border: 1px solid #EAB308; }

    /* Secciones */
    .section-title {
        color: #FAFAFA;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 24px 0 12px 0;
        padding-left: 12px;
        border-left: 4px solid #00D4AA;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0B0F16;
        border-right: 1px solid #2A3142;
    }

    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA 0%, #00A88A 100%);
        color: #0E1117;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.35);
    }

    /* DataFrame */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2A3142;
    }

    /* Ocultar branding de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCIONES CORE (tu lógica del notebook)
# =========================================================
DEFAULT_TICKERS = ["NVDA", "MU", "AMD", "INTC", "AVGO", "GOOG", "META", "MSFT", "ORCL"]
FEATURES = ["Slope_SMA_40", "Slope_SMA_80", "Slope_SMA_160", "RSI", "Relative_Range"]
LABEL_MAP = {2: "ALCISTA", 1: "BAJISTA", 0: "LATERAL"}


@st.cache_data(show_spinner=False, ttl=3600)
def download_data(tickers, period="5y"):
    data = yf.download(tickers, period=period, interval="1d",
                       group_by="ticker", progress=False, auto_adjust=True)
    return data


def clean_stock_data(df):
    data = df.dropna().copy()
    data = data[(data["Volume"] > 0) & (data["High"] != data["Low"])].copy()
    return data


def create_features_and_target(df):
    d = df.copy()
    d["SMA_40"] = d["Close"].rolling(40).mean()
    d["SMA_80"] = d["Close"].rolling(80).mean()
    d["SMA_160"] = d["Close"].rolling(160).mean()

    d["Slope_SMA_40"] = d["SMA_40"].pct_change(5)
    d["Slope_SMA_80"] = d["SMA_80"].pct_change(5)
    d["Slope_SMA_160"] = d["SMA_160"].pct_change(5)

    delta = d["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI"] = 100 - (100 / (1 + rs))

    d["Relative_Range"] = (d["High"] - d["Low"]) / d["Close"]

    future_return = d["Close"].shift(-3).pct_change(3)
    d["Target"] = 0
    d.loc[future_return > 0.01, "Target"] = 2
    d.loc[future_return < -0.01, "Target"] = 1

    d = d.dropna().copy()
    return d


@st.cache_data(show_spinner=False, ttl=3600)
def process_all_tickers(tickers, period="5y"):
    raw = download_data(tuple(tickers), period)
    processed = {}
    for t in tickers:
        try:
            sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            cleaned = clean_stock_data(sub)
            if len(cleaned) > 200:
                processed[t] = create_features_and_target(cleaned)
        except Exception as e:
            st.warning(f"No se pudo procesar {t}: {e}")
    return processed


def train_and_evaluate(df_stock):
    X = df_stock[FEATURES]
    y = df_stock["Target"]
    split = int(len(df_stock) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    knn = KNeighborsClassifier(n_neighbors=50)
    knn.fit(X_tr, y_tr)
    acc_knn = accuracy_score(y_te, knn.predict(X_te))

    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                        random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    acc_xgb = accuracy_score(y_te, xgb.predict(X_te))

    acc_majority = y_te.value_counts(normalize=True).iloc[0]

    return {
        "knn_acc": acc_knn,
        "xgb_acc": acc_xgb,
        "majority_acc": acc_majority,
        "best": "KNN" if acc_knn >= acc_xgb else "XGBoost",
        "xgb_model": xgb,
    }


def predict_signal(df_stock):
    X_all = df_stock[FEATURES]
    y_all = df_stock["Target"]
    model = KNeighborsClassifier(n_neighbors=50)
    model.fit(X_all, y_all)

    latest = X_all.iloc[[-1]]
    pred = model.predict(latest)[0]
    probs = model.predict_proba(latest)[0]
    return LABEL_MAP[pred], probs.max(), df_stock.index[-1]


# =========================================================
# HEADER PRINCIPAL
# =========================================================
st.markdown("""
<div class="hero">
    <h1>📈 AI Stock Predictor <span class="accent">Pro</span></h1>
    <p>Predicción bursátil con Machine Learning · KNN y XGBoost sobre indicadores técnicos</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    tickers_input = st.text_area(
        "Tickers a analizar (separados por coma)",
        value=", ".join(DEFAULT_TICKERS),
        height=80,
        help="Ingresa símbolos bursátiles válidos de Yahoo Finance."
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    period = st.selectbox(
        "Periodo histórico",
        options=["1y", "2y", "5y", "10y"],
        index=2,
        help="Ventana de datos para entrenar los modelos."
    )

    st.markdown("---")
    st.markdown("### 🎛️ Opciones")
    show_details = st.checkbox("Mostrar detalles técnicos", value=False)

    st.markdown("---")
    run_btn = st.button("🚀 Ejecutar análisis", use_container_width=True)

    st.markdown("---")
    st.caption("💡 **Tip:** El modelo KNN con k=50 ha demostrado mayor consistencia en distintos tickers.")


# =========================================================
# FLUJO PRINCIPAL
# =========================================================
if run_btn or "analyzed" not in st.session_state:
    if not tickers:
        st.error("⚠️ Debes ingresar al menos un ticker.")
        st.stop()

    with st.spinner("🔄 Descargando datos de mercado y entrenando modelos..."):
        processed = process_all_tickers(tickers, period)

    if not processed:
        st.error("❌ No se pudieron procesar los tickers ingresados. Verifica los símbolos.")
        st.stop()

    # Entrenamiento y predicción
    results = []
    signals = []
    for t, df_s in processed.items():
        eval_res = train_and_evaluate(df_s)
        signal, conf, last_date = predict_signal(df_s)

        results.append({
            "Ticker": t,
            "Clase Mayoritaria": f"{eval_res['majority_acc']:.2%}",
            "Precisión KNN": f"{eval_res['knn_acc']:.2%}",
            "Precisión XGBoost": f"{eval_res['xgb_acc']:.2%}",
            "Mejor Modelo": eval_res["best"],
            "_knn": eval_res["knn_acc"],
            "_xgb": eval_res["xgb_acc"],
            "_maj": eval_res["majority_acc"],
        })
        signals.append({
            "Ticker": t,
            "Señal": signal,
            "Confianza": conf,
            "Fecha": last_date.strftime("%Y-%m-%d"),
        })

    st.session_state["analyzed"] = True
    st.session_state["processed"] = processed
    st.session_state["results"] = results
    st.session_state["signals"] = signals


# =========================================================
# RENDERIZADO DE RESULTADOS
# =========================================================
if "results" in st.session_state:
    processed = st.session_state["processed"]
    results = st.session_state["results"]
    signals = st.session_state["signals"]

    signals_df = pd.DataFrame(signals)
    results_df = pd.DataFrame(results)

    # ---------------- KPIs GLOBALES ----------------
    st.markdown('<div class="section-title">📊 Resumen Ejecutivo</div>', unsafe_allow_html=True)

    avg_knn = results_df["_knn"].mean()
    avg_xgb = results_df["_xgb"].mean()
    avg_maj = results_df["_maj"].mean()
    best_overall = "KNN" if avg_knn >= avg_xgb else "XGBoost"
    n_bull = (signals_df["Señal"] == "ALCISTA").sum()
    n_bear = (signals_df["Señal"] == "BAJISTA").sum()
    n_flat = (signals_df["Señal"] == "LATERAL").sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Tickers analizados</div>
            <div class="kpi-value">{len(processed)}</div>
            <div class="kpi-sub">Datos hasta {signals_df['Fecha'].max()}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Precisión promedio KNN</div>
            <div class="kpi-value">{avg_knn:.1%}</div>
            <div class="kpi-sub">Baseline: {avg_maj:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Precisión promedio XGBoost</div>
            <div class="kpi-value">{avg_xgb:.1%}</div>
            <div class="kpi-sub">Mejor global: {best_overall}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Distribución de señales</div>
            <div class="kpi-value">{n_bull} 🟢 / {n_bear} 🔴</div>
            <div class="kpi-sub">{n_flat} señales laterales</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- SEÑALES ACTUALES ----------------
    st.markdown('<div class="section-title">🎯 Señales de Trading Actuales (Modelo KNN)</div>', unsafe_allow_html=True)

    def signal_badge(s):
        if s == "ALCISTA":
            return '<span class="badge badge-bull">🟢 ALCISTA</span>'
        elif s == "BAJISTA":
            return '<span class="badge badge-bear">🔴 BAJISTA</span>'
        return '<span class="badge badge-flat">🟡 LATERAL</span>'

    cols = st.columns(min(3, len(signals_df)))
    for i, row in signals_df.iterrows():
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.4rem; font-weight:800; color:#FAFAFA;">{row['Ticker']}</span>
                    {signal_badge(row['Señal'])}
                </div>
                <div style="margin-top:10px; color:#9CA3AF; font-size:0.85rem;">Confianza del modelo</div>
                <div style="font-size:1.5rem; font-weight:700; color:#00D4AA;">{row['Confianza']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- TABLA COMPARATIVA ----------------
    st.markdown('<div class="section-title">📋 Comparativa de Modelos por Ticker</div>', unsafe_allow_html=True)

    display_df = results_df[["Ticker", "Clase Mayoritaria", "Precisión KNN",
                             "Precisión XGBoost", "Mejor Modelo"]].copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ---------------- GRÁFICOS ----------------
    st.markdown('<div class="section-title">📉 Comparación Visual de Precisión</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Clase Mayoritaria",
        x=results_df["Ticker"], y=results_df["_maj"],
        marker_color="#6B7280",
        text=[f"{v:.1%}" for v in results_df["_maj"]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="KNN",
        x=results_df["Ticker"], y=results_df["_knn"],
        marker_color="#00D4AA",
        text=[f"{v:.1%}" for v in results_df["_knn"]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="XGBoost",
        x=results_df["Ticker"], y=results_df["_xgb"],
        marker_color="#EF4444",
        text=[f"{v:.1%}" for v in results_df["_xgb"]],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(title="Precisión", tickformat=".0%", gridcolor="#2A3142"),
        xaxis=dict(gridcolor="#2A3142"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(color="#FAFAFA"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- DETALLE POR TICKER ----------------
    st.markdown('<div class="section-title">🔍 Análisis Detallado por Ticker</div>', unsafe_allow_html=True)

    selected = st.selectbox("Selecciona un ticker para explorar:", list(processed.keys()))
    df_sel = processed[selected]

    col_a, col_b = st.columns([2, 1])

    with col_a:
        price_fig = go.Figure()
        price_fig.add_trace(go.Candlestick(
            x=df_sel.index,
            open=df_sel["Open"], high=df_sel["High"],
            low=df_sel["Low"], close=df_sel["Close"],
            name="Precio",
            increasing_line_color="#00D4AA",
            decreasing_line_color="#EF4444",
        ))
        price_fig.add_trace(go.Scatter(
            x=df_sel.index, y=df_sel["SMA_40"],
            mode="lines", name="SMA 40", line=dict(color="#3B82F6", width=1.5)
        ))
        price_fig.add_trace(go.Scatter(
            x=df_sel.index, y=df_sel["SMA_160"],
            mode="lines", name="SMA 160", line=dict(color="#F59E0B", width=1.5)
        ))
        price_fig.update_layout(
            title=f"{selected} · Precio + Medias Móviles",
            template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=440, xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=50, b=20),
            font=dict(color="#FAFAFA"),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        st.plotly_chart(price_fig, use_container_width=True)

    with col_b:
        # Importancia de features con XGBoost entrenado completo
        X_all = df_sel[FEATURES]
        y_all = df_sel["Target"]
        xgb_full = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                 random_state=42, verbosity=0)
        xgb_full.fit(X_all, y_all)
        importances = pd.Series(xgb_full.feature_importances_, index=FEATURES).sort_values()

        imp_fig = px.bar(
            x=importances.values, y=importances.index, orientation="h",
            title=f"Importancia de Features · {selected}",
            labels={"x": "Peso", "y": ""},
        )
        imp_fig.update_traces(marker_color="#00D4AA", text=[f"{v:.2f}" for v in importances.values],
                              textposition="outside")
        imp_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
            height=440, margin=dict(l=20, r=20, t=50, b=20),
            font=dict(color="#FAFAFA"),
        )
        st.plotly_chart(imp_fig, use_container_width=True)

    # ---------------- RSI ----------------
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=df_sel.index, y=df_sel["RSI"], mode="lines",
        line=dict(color="#A855F7", width=2), name="RSI"
    ))
    rsi_fig.add_hline(y=70, line_dash="dash", line_color="#EF4444", annotation_text="Sobrecompra (70)")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="#00D4AA", annotation_text="Sobreventa (30)")
    rsi_fig.update_layout(
        title=f"{selected} · RSI (14)",
        template="plotly_dark",
        paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        font=dict(color="#FAFAFA"),
    )
    st.plotly_chart(rsi_fig, use_container_width=True)

    # ---------------- DETALLES TÉCNICOS ----------------
    if show_details:
        with st.expander("🧠 Ver distribución del Target y datos crudos"):
            st.markdown(f"**Distribución de clases del Target para {selected}:**")
            dist = df_sel["Target"].value_counts(normalize=True).rename(
                index=LABEL_MAP).round(3)
            st.dataframe(dist, use_container_width=True)

            st.markdown("**Últimas 10 velas procesadas:**")
            st.dataframe(df_sel.tail(10), use_container_width=True)

    # ---------------- FOOTER ----------------
    st.markdown("---")
    st.caption(
        "⚠️ **Aviso:** Este análisis es con fines educativos y no constituye asesoría financiera. "
        "El trading conlleva riesgo de pérdida de capital. Los modelos se basan en patrones históricos "
        "y no garantizan resultados futuros."
    )
    st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
