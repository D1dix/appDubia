import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dubai PropTech — IA Screening",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Sans:wght@300;400;500&display=swap');
    
    .main { background-color: #0a0a0f; }
    .block-container { padding: 0 2rem 2rem 2rem; background: #0a0a0f; }
    
    h1 { font-family: 'Cormorant Garamond', serif !important; font-weight: 300 !important;
         letter-spacing: 6px !important; color: #c9a84c !important; text-transform: uppercase; font-size: 2.2rem !important; }
    
    .stSlider label, .stSelectbox label { 
        font-size: 0.7rem !important; letter-spacing: 2px !important; 
        color: #8a8070 !important; text-transform: uppercase !important; font-family: 'DM Sans' !important; }
    
    .metric-card {
        background: #13131e; border: 1px solid #c9a84c33; padding: 1rem 1.2rem;
        margin: 0.3rem 0; font-family: 'DM Sans', sans-serif;
    }
    .metric-label { font-size: 0.65rem; letter-spacing: 2px; color: #6a6055; text-transform: uppercase; margin-bottom: 4px; }
    .metric-val { font-size: 1.1rem; font-weight: 500; }
    
    .result-box {
        background: #0d0d14; border: 1px solid #c9a84c33; padding: 1.5rem;
        text-align: center; font-family: 'DM Sans', sans-serif; margin: 1rem 0;
    }
    .prob-num { font-family: 'Cormorant Garamond', serif; font-size: 4rem; font-weight: 300; line-height: 1; }
    .verdict { font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase; padding: 4px 14px; display: inline-block; margin-top: 8px; }
    
    .ok   { background: #4caf8222; color: #4caf82; border: 1px solid #4caf8244; }
    .no   { background: #e24b4a22; color: #e24b4a; border: 1px solid #e24b4a44; }
    .maybe{ background: #c9a84c22; color: #c9a84c; border: 1px solid #c9a84c44; }
    
    .crit-row { display: flex; justify-content: space-between; align-items: center;
                padding: 8px 0; border-bottom: 1px solid #ffffff08; font-size: 0.8rem; }
    .crit-name { color: #8a8070; font-size: 0.65rem; letter-spacing: 1px; text-transform: uppercase; }
    
    div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
    .stButton button {
        width: 100%; background: #c9a84c !important; color: #0a0a0f !important;
        border: none !important; padding: 0.8rem !important;
        font-family: 'DM Sans' !important; font-size: 0.7rem !important;
        font-weight: 500 !important; letter-spacing: 3px !important;
        text-transform: uppercase !important; border-radius: 0 !important;
    }
    .header-bar {
        background: #0d0d14; border-bottom: 1px solid #c9a84c33;
        padding: 1rem 2rem; margin: -2rem -2rem 2rem -2rem;
        display: flex; justify-content: space-between; align-items: center;
    }
    .tag { font-size: 0.6rem; letter-spacing: 2px; color: #c9a84c;
           border: 1px solid #c9a84c44; padding: 3px 10px; text-transform: uppercase; }
    
    section[data-testid="stSidebar"] { display: none; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    modelo = joblib.load("modelo_dubai_xgb.pkl")
    with open("features_config.json") as f:
        cfg = json.load(f)
    return modelo, cfg

modelo, cfg = load_model()
log_features = cfg['log_features']
num_features = cfg['num_features']
cat_nominal  = cfg['cat_nominal']
cat_ordinal  = cfg['cat_ordinal']
todas = log_features + num_features + cat_nominal + cat_ordinal


ZONAS = [
    {"id": "downtown", "name": "Downtown Dubai",    "lat": 25.1972, "lng": 55.2744, "min": 400000, "desc": "Burj Khalifa · Lujo absoluto"},
    {"id": "marina",   "name": "Dubai Marina",      "lat": 25.0804, "lng": 55.1404, "min": 250000, "desc": "Frente al mar · Vida cosmopolita"},
    {"id": "palm",     "name": "Palm Jumeirah",      "lat": 25.1124, "lng": 55.1390, "min": 800000, "desc": "Isla artificial · Ultra premium"},
    {"id": "business", "name": "Business Bay",       "lat": 25.1862, "lng": 55.2914, "min": 300000, "desc": "Canal de Dubai · Negocios"},
    {"id": "creek",    "name": "Creek Harbour",      "lat": 25.2198, "lng": 55.3524, "min": 200000, "desc": "Nuevo desarrollo · Gran potencial"},
    {"id": "jvc",      "name": "Jumeirah Village",   "lat": 25.0549, "lng": 55.2066, "min": 120000, "desc": "Accesible · Comunidad familiar"},
    {"id": "difc",     "name": "DIFC",               "lat": 25.2124, "lng": 55.2814, "min": 350000, "desc": "Distrito financiero internacional"},
    {"id": "jbr",      "name": "JBR Beach",          "lat": 25.0759, "lng": 55.1319, "min": 280000, "desc": "The Walk · Primera línea playa"},
]


def feature_engineering(d):
    d['capacidad_pago_mensual'] = (d['ingresos_anuales'] * 0.30 / 12) - (d['deudas_totales'] / 12)
    d['tasa_ahorro']            = d['dinero_banco'] / max(d['ingresos_anuales'], 1)
    d['empleo_estable']         = int(d.get('meses_empresa_actual', 36) >= 24)
    d['diversificacion_inv']    = (int(d.get('invertido_bolsa', 0) > 0) +
                                   int(d.get('invertido_cripto', 0) > 0) +
                                   int(d.get('invertido_propiedades', 0) > 0))
    d['patrimonio_liquido']     = (d['dinero_banco'] +
                                   d.get('invertido_bolsa', 0) * 0.95 +
                                   d.get('invertido_cripto', 0) * 0.90)
    d['ratio_affordability']    = d.get('presupuesto_max', d['patrimonio_neto'] * 0.8) / max(d['patrimonio_neto'], 1)
    d['carga_familiar']         = d.get('dependientes_menores', 0) / max(d.get('num_integrantes_familia', 1), 1)
    edad = d.get('edad', 38)
    if edad < 32:    d['grupo_edad'] = 'Joven'
    elif edad < 45:  d['grupo_edad'] = 'Adulto_Medio'
    elif edad < 58:  d['grupo_edad'] = 'Adulto_Mayor'
    else:            d['grupo_edad'] = 'Pre_Jubilacion'
    return d


def predict(data):
    defaults = {
        'presupuesto_max': data.get('patrimonio_neto', 0) * 0.8,
        'invertido_bolsa': 0, 'invertido_cripto': 0, 'invertido_propiedades': 0,
        'renta_pasiva': 0, 'gastos_mensuales': data.get('ingresos_anuales', 0) * 0.03,
        'anos_experiencia': 8, 'meses_empresa_actual': 36,
        'num_integrantes_familia': 3, 'dependientes_menores': 1,
        'genero': 'Masculino', 'estado_civil': 'Soltero',
        'tipo_contrato': 'Indefinido', 'motivo_compra': 'Inversion',
        'pais_origen': 'India', 'zona_preferida': 'Downtown Dubai',
        'profesion_grupo': 'Otras', 'sector_grupo': 'Otros',
    }
    for k, v in defaults.items():
        if k not in data:
            data[k] = v

    data = feature_engineering(data)
    df = pd.DataFrame([data])
    for col in todas:
        if col not in df.columns:
            df[col] = 0

    for col in log_features:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    X = df[todas]
    prob = float(modelo.predict_proba(X)[0][1])
    pred = int(modelo.predict(X)[0])
    return prob, pred


def build_map(prob, presupuesto):
    m = folium.Map(
        location=[25.15, 55.22], zoom_start=12,
        tiles='CartoDB dark_matter',
        attr='CartoDB'
    )

    for z in ZONAS:
        asequible = presupuesto >= z['min'] and prob >= 0.35
        perfecto  = presupuesto >= z['min'] * 1.2 and prob >= 0.60

        color  = '#4caf82' if perfecto else '#c9a84c' if asequible else '#e24b4a'
        fill   = 0.25 if perfecto else 0.18 if asequible else 0.10
        radius = 900 if perfecto else 700 if asequible else 500
        fit    = '✓ Zona ideal' if perfecto else '~ Asequible' if asequible else '✗ Fuera de rango'

        folium.Circle(
            location=[z['lat'], z['lng']],
            radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=fill, weight=2, opacity=0.9,
            tooltip=folium.Tooltip(f"<b style='font-size:13px'>{z['name']}</b><br/>"
                                   f"<span style='font-size:11px;color:#aaa'>{z['desc']}</span><br/>"
                                   f"<span style='font-size:11px'>Precio mínimo: ${z['min']:,}</span><br/>"
                                   f"<b style='color:{color};font-size:12px'>{fit}</b>",
                                   sticky=True)
        ).add_to(m)

        folium.CircleMarker(
            location=[z['lat'], z['lng']],
            radius=5, color=color,
            fill=True, fill_color=color, fill_opacity=1,
            tooltip=z['name']
        ).add_to(m)

        folium.map.Marker(
            [z['lat'] + 0.007, z['lng']],
            icon=folium.DivIcon(
                html=f"<div style='font-family:DM Sans,sans-serif;font-size:10px;"
                     f"color:{color};font-weight:500;letter-spacing:1px;"
                     f"text-shadow:0 0 4px #000;white-space:nowrap'>{z['name']}</div>",
                icon_size=(140, 20), icon_anchor=(70, 0)
            )
        ).add_to(m)

    return m


# ─────────────────────────────── HEADER ──────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div style="font-family:'Cormorant Garamond',serif;font-size:1.4rem;font-weight:300;
              letter-spacing:5px;color:#c9a84c;text-transform:uppercase">
    DUBAI <b style="color:#e8e0d0;font-weight:600">PROPTECH</b>
    <span style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#6a6055;
                 letter-spacing:1px;margin-left:12px">— Screening IA</span>
  </div>
  <div class="tag">XGBoost Optimizado · AUC 0.9697</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────── LAYOUT ──────────────────────────────────────
col_form, col_results = st.columns([1, 1.6], gap="large")

with col_form:
    st.markdown("<p style='font-size:0.65rem;letter-spacing:3px;color:#c9a84c;"
                "text-transform:uppercase;margin-bottom:1rem'>Perfil financiero del cliente</p>",
                unsafe_allow_html=True)

    ingresos  = st.slider("Ingresos anuales (USD)",  20000,  500000, 120000, 5000,  format="$%d")
    patrimonio= st.slider("Patrimonio neto (USD)",        0,  600000,  80000, 5000,  format="$%d")
    deudas    = st.slider("Deudas totales (USD)",         0,  300000,  30000, 5000,  format="$%d")
    banco     = st.slider("Dinero en banco (USD)",        0,  200000,  40000, 5000,  format="$%d")
    score     = st.slider("Score crediticio",            300,     850,    700,   10)
    edad      = st.slider("Edad",                         22,      70,     38,    1)

    col_a, col_b = st.columns(2)
    with col_a:
        cat_prof = st.selectbox("Categoría profesional",
                                ["Mid-Level", "Senior", "Director", "C-Level"], index=1)
    with col_b:
        interes  = st.selectbox("Interés en compra", ["Bajo", "Medio", "Alto"], index=1)
        nivel_edu= st.selectbox("Nivel educativo",
                                ["Bachillerato", "Licenciatura", "Master", "PhD"], index=1)

    analizar = st.button("Analizar viabilidad y ver zonas en Dubai")


# ─────────────────────────────── PREDICTION ──────────────────────────────────
if analizar or 'resultado' in st.session_state:

    if analizar:
        input_data = {
            'ingresos_anuales': ingresos,
            'patrimonio_neto':  patrimonio,
            'deudas_totales':   deudas,
            'dinero_banco':     banco,
            'score_crediticio': score,
            'edad':             edad,
            'categoria_profesional': cat_prof,
            'interes_compra_dubai':  interes,
            'nivel_educativo':       nivel_edu,
            'presupuesto_max':       patrimonio * 0.8,
        }
        prob, pred = predict(input_data)
        ratio = (deudas / max(ingresos, 1)) * 100
        c1 = patrimonio >= 30000
        c2 = ratio < 40
        c3 = score >= 650

        st.session_state['resultado'] = {
            'prob': prob, 'pred': pred, 'ratio': ratio,
            'c1': c1, 'c2': c2, 'c3': c3,
            'presupuesto': patrimonio * 0.6 + ingresos * 0.3,
            'ingresos': ingresos, 'patrimonio': patrimonio,
            'deudas': deudas, 'score': score
        }

    r = st.session_state['resultado']
    prob  = r['prob']
    pct   = round(prob * 100, 1)
    color = '#4caf82' if pct >= 60 else '#c9a84c' if pct >= 35 else '#e24b4a'
    vtext = 'Cliente viable' if pct >= 60 else 'Perfil borderline' if pct >= 35 else 'No viable'
    vcls  = 'ok' if pct >= 60 else 'maybe' if pct >= 35 else 'no'

    with col_results:
        # Resultado principal
        st.markdown(f"""
        <div class="result-box">
          <div style="font-size:0.65rem;letter-spacing:3px;color:#6a6055;text-transform:uppercase;margin-bottom:8px">
            Probabilidad de viabilidad · XGBoost
          </div>
          <div class="prob-num" style="color:{color}">{pct}%</div>
          <div><span class="verdict {vcls}">{vtext}</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Criterios
        c_icon = lambda ok: ("✓" if ok else "✗")
        c_col  = lambda ok: ("#4caf82" if ok else "#e24b4a")

        st.markdown(f"""
        <div style="background:#13131e;border:1px solid #c9a84c22;padding:1rem;margin-bottom:1rem">
          <div style="font-size:0.65rem;letter-spacing:2px;color:#c9a84c;text-transform:uppercase;margin-bottom:10px">
            Criterios del modelo
          </div>
          <div class="crit-row">
            <span class="crit-name">Patrimonio neto ≥ $30,000</span>
            <span style="color:{c_col(r['c1'])};font-weight:500">${r['patrimonio']:,.0f} {c_icon(r['c1'])}</span>
          </div>
          <div class="crit-row">
            <span class="crit-name">Ratio deuda/ingreso &lt; 40%</span>
            <span style="color:{c_col(r['c2'])};font-weight:500">{r['ratio']:.1f}% {c_icon(r['c2'])}</span>
          </div>
          <div class="crit-row" style="border-bottom:none">
            <span class="crit-name">Score crediticio ≥ 650</span>
            <span style="color:{c_col(r['c3'])};font-weight:500">{r['score']} pts {c_icon(r['c3'])}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Mapa
        st.markdown("<p style='font-size:0.65rem;letter-spacing:2px;color:#c9a84c;"
                    "text-transform:uppercase;margin-bottom:0.5rem'>Zonas de Dubai — pasa el cursor sobre cada zona</p>",
                    unsafe_allow_html=True)

        m = build_map(prob, r['presupuesto'])
        st_folium(m, width=None, height=400, returned_objects=[])

        # Leyenda
        st.markdown("""
        <div style="display:flex;gap:20px;margin-top:0.5rem;font-size:0.65rem;
                    letter-spacing:1px;text-transform:uppercase;color:#6a6055">
          <span style="color:#4caf82">● Zona ideal</span>
          <span style="color:#c9a84c">● Asequible</span>
          <span style="color:#e24b4a">● Fuera de rango</span>
        </div>
        """, unsafe_allow_html=True)

else:
    with col_results:
        st.markdown("""
        <div style="height:500px;display:flex;align-items:center;justify-content:center;
                    border:1px solid #c9a84c22;background:#0d0d14;text-align:center">
          <div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:2rem;
                        color:#c9a84c33;letter-spacing:4px;text-transform:uppercase">Dubai</div>
            <div style="font-size:0.65rem;letter-spacing:2px;color:#3a3530;
                        text-transform:uppercase;margin-top:8px">
              Introduce datos y analiza para ver el mapa
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
