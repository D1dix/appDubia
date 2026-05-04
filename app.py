import streamlit as st
import joblib, json, numpy as np, pandas as pd
import folium, plotly.graph_objects as go
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dubai PropTech · AI Screening", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

# ═══════════════════════════════════════════════════════════════════
# CSS GLOBAL — paleta #0E0E0E · #D4AF37 · #10B981 · #B91C1C
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0E0E0E; }
.block-container { padding: 0 !important; max-width: 100% !important; background: #0E0E0E; }
section[data-testid="stSidebar"], footer, header { display: none !important; }
#MainMenu { display: none !important; }

/* ── WRAPPER CENTRAL ── */
.wrap { max-width: 1400px; margin: 0 auto; padding: 0 24px 40px 24px; }

/* ── HEADER ── */
.hdr {
  background: linear-gradient(135deg, #0E0E0E 0%, #1A1400 100%);
  border-bottom: 1px solid rgba(212,175,55,0.25);
  padding: 18px 32px; display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 0;
}
.hdr-logo { font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; font-weight: 300; letter-spacing: 6px; color: #D4AF37; text-transform: uppercase; }
.hdr-logo b { color: #fff; font-weight: 700; }
.hdr-pills { display: flex; gap: 10px; align-items: center; }
.pill {
  font-size: 0.58rem; letter-spacing: 2px; text-transform: uppercase; font-weight: 600;
  padding: 5px 12px; border: 1px solid rgba(212,175,55,0.4); color: #D4AF37;
  background: rgba(212,175,55,0.08); border-radius: 20px;
  animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(212,175,55,0.3)} 50%{box-shadow:0 0 0 6px rgba(212,175,55,0)} }

/* ── SECTION TITLES ── */
.sec-title {
  font-family: 'Cormorant Garamond', serif; font-size: 0.7rem;
  letter-spacing: 4px; color: #D4AF37; text-transform: uppercase;
  margin: 20px 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(212,175,55,0.2);
}

/* ── PRESET CARDS ── */
.presets { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; margin: 18px 0 22px 0; }
.preset-card {
  background: #1A1A1A; border: 1px solid rgba(212,175,55,0.18); border-radius: 14px;
  padding: 14px 12px; cursor: pointer; transition: all 0.2s; text-align: center;
}
.preset-card:hover { border-color: #D4AF37; background: #1e1c14; transform: translateY(-2px); box-shadow: 0 8px 24px rgba(212,175,55,0.12); }
.preset-icon { font-size: 1.6rem; margin-bottom: 6px; }
.preset-name { font-size: 0.72rem; font-weight: 600; color: #e8e0d0; letter-spacing: 0.5px; margin-bottom: 3px; }
.preset-tag { font-size: 0.58rem; color: #6a6055; letter-spacing: 0.5px; }

/* ── PANEL CARDS ── */
.panel {
  background: #1A1A1A; border: 1px solid rgba(212,175,55,0.18); border-radius: 16px;
  padding: 22px 20px; margin-bottom: 14px;
}

/* ── SLIDERS ── */
.stSlider label { font-size: 0.62rem !important; letter-spacing: 2px !important; color: #6a6055 !important; text-transform: uppercase !important; font-family: 'Inter' !important; }
.stSlider > div > div > div { background: #2a2820 !important; border-radius: 4px !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg, #D4AF37, #e0c860) !important; border-radius: 4px !important; }
[data-testid="stSliderThumb"] { background: #D4AF37 !important; border: 2px solid #fff !important; box-shadow: 0 0 10px rgba(212,175,55,0.6) !important; width: 18px !important; height: 18px !important; }

/* ── SELECTBOX ── */
.stSelectbox label { font-size: 0.62rem !important; letter-spacing: 2px !important; color: #6a6055 !important; text-transform: uppercase !important; }
.stSelectbox > div > div { background: #111 !important; border: 1px solid rgba(212,175,55,0.2) !important; border-radius: 8px !important; color: #e8e0d0 !important; }

/* ── BOTONES ── */
.stButton > button {
  background: linear-gradient(135deg, #D4AF37, #b8941f) !important; color: #0E0E0E !important;
  border: none !important; border-radius: 10px !important; font-weight: 700 !important;
  letter-spacing: 2px !important; font-size: 0.68rem !important; text-transform: uppercase !important;
  padding: 12px 20px !important; width: 100% !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(212,175,55,0.3) !important; }

/* ── VERDICT BADGES ── */
.badge { font-size: 0.62rem; letter-spacing: 3px; text-transform: uppercase; padding: 6px 16px; border-radius: 20px; display: inline-block; font-weight: 600; margin-top: 10px; }
.badge-ok { background: rgba(16,185,129,0.15); color: #10B981; border: 1px solid rgba(16,185,129,0.4); }
.badge-no { background: rgba(185,28,28,0.15); color: #ef4444; border: 1px solid rgba(185,28,28,0.4); }
.badge-maybe { background: rgba(212,175,55,0.15); color: #D4AF37; border: 1px solid rgba(212,175,55,0.4); }

/* ── CRITERIA ROWS ── */
.crit-row { display:flex; justify-content:space-between; align-items:center; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:0.78rem; }
.crit-row:last-child { border-bottom:none; }
.crit-label { color:#8a8070; font-size:0.62rem; letter-spacing:1px; text-transform:uppercase; }

/* ── ADVICE CARD ── */
.advice { background: rgba(185,28,28,0.08); border: 1px solid rgba(185,28,28,0.3); border-radius: 12px; padding: 16px; margin-top: 12px; }
.advice-title { font-size:0.65rem; letter-spacing:2px; color:#ef4444; text-transform:uppercase; margin-bottom:10px; font-weight:600; }
.advice-item { display:flex; align-items:flex-start; gap:8px; margin-bottom:8px; font-size:0.78rem; color:#c0b090; }
.advice-icon { color:#D4AF37; margin-top:1px; }

/* ── ZONE CARDS GRID ── */
.zones-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
.zone-card { background:#111; border-left:3px solid; border-radius:0 8px 8px 0; padding:10px 12px; }
.zone-name { font-size:0.75rem; font-weight:600; margin-bottom:2px; }
.zone-desc { font-size:0.6rem; color:#6a6055; }
.zone-price { font-size:0.62rem; margin-top:4px; font-weight:500; }

/* ── EXPLAINABILITY ── */
.feat-bar-wrap { margin: 6px 0; }
.feat-label { font-size:0.62rem; color:#8a8070; text-transform:uppercase; letter-spacing:1px; display:flex; justify-content:space-between; margin-bottom:3px; }
.feat-bar-bg { background:#1a1a1a; border-radius:4px; height:6px; }
.feat-bar { height:6px; border-radius:4px; background:linear-gradient(90deg,#D4AF37,#e0c860); }

/* ── FOOTER ── */
.footer { text-align:center; padding:20px; font-size:0.6rem; color:#3a3530; letter-spacing:2px; text-transform:uppercase; border-top:1px solid rgba(212,175,55,0.1); margin-top:20px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# PERFILES PREDEFINIDOS — edita aquí para cambiarlos
# ═══════════════════════════════════════════════════════════════════
PRESETS = {
    "🏆 Whale Investor":      {"icon":"🏆","tag":"Ticket alto · Zonas ultra premium","ingresos":850000,"patrimonio":4500000,"deudas":200000,"banco":800000,"score":820,"edad":55,"cat":"C-Level","interes":"Alto","edu":"PhD"},
    "💼 Expat Ejecutivo":     {"icon":"💼","tag":"Perfil consolidado · Múltiples zonas","ingresos":280000,"patrimonio":650000,"deudas":120000,"banco":180000,"score":760,"edad":42,"cat":"Director","interes":"Alto","edu":"Master"},
    "🚀 Tech Founder":        {"icon":"🚀","tag":"Alto potencial · En expansión","ingresos":180000,"patrimonio":400000,"deudas":80000,"banco":120000,"score":720,"edad":34,"cat":"Senior","interes":"Alto","edu":"Master"},
    "👨‍👩‍👧 Familia Profesional": {"icon":"👨‍👩‍👧","tag":"Estable · Zonas residenciales","ingresos":140000,"patrimonio":220000,"deudas":90000,"banco":60000,"score":700,"edad":38,"cat":"Senior","interes":"Medio","edu":"Licenciatura"},
    "🌱 Joven Profesional":   {"icon":"🌱","tag":"Perfil borderline · JVC viable","ingresos":75000,"patrimonio":40000,"deudas":30000,"banco":15000,"score":680,"edad":28,"cat":"Mid-Level","interes":"Medio","edu":"Licenciatura"},
}

# ═══════════════════════════════════════════════════════════════════
# ZONAS DE DUBAI
# ═══════════════════════════════════════════════════════════════════
ZONAS = [
    {"id":"downtown","name":"Downtown Dubai", "lat":25.1972,"lng":55.2744,"min":400000,"m2":22000,"desc":"Burj Khalifa · Lujo absoluto","emoji":"🏙️"},
    {"id":"marina",  "name":"Dubai Marina",   "lat":25.0804,"lng":55.1404,"min":250000,"m2":16000,"desc":"Frente al mar · Cosmopolita","emoji":"⛵"},
    {"id":"palm",    "name":"Palm Jumeirah",  "lat":25.1124,"lng":55.1390,"min":800000,"m2":35000,"desc":"Isla artificial · Ultra premium","emoji":"🌴"},
    {"id":"business","name":"Business Bay",   "lat":25.1862,"lng":55.2914,"min":300000,"m2":18000,"desc":"Canal de Dubai · Negocios","emoji":"💼"},
    {"id":"creek",   "name":"Creek Harbour",  "lat":25.2198,"lng":55.3524,"min":200000,"m2":14000,"desc":"Nuevo desarrollo · Gran potencial","emoji":"🌊"},
    {"id":"jvc",     "name":"Jumeirah Village","lat":25.0549,"lng":55.2066,"min":120000,"m2":9000, "desc":"Accesible · Comunidad familiar","emoji":"🏡"},
    {"id":"difc",    "name":"DIFC",           "lat":25.2124,"lng":55.2814,"min":350000,"m2":20000,"desc":"Distrito financiero internacional","emoji":"🏦"},
    {"id":"jbr",     "name":"JBR Beach",      "lat":25.0759,"lng":55.1319,"min":280000,"m2":17000,"desc":"The Walk · Primera línea playa","emoji":"🏖️"},
]

# ═══════════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ═══════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════
# LÓGICA DEL MODELO
# ═══════════════════════════════════════════════════════════════════
def feature_engineering(d):
    d['capacidad_pago_mensual'] = (d['ingresos_anuales']*0.30/12)-(d['deudas_totales']/12)
    d['tasa_ahorro']            = d['dinero_banco']/max(d['ingresos_anuales'],1)
    d['empleo_estable']         = int(d.get('meses_empresa_actual',36)>=24)
    d['diversificacion_inv']    = int(d.get('invertido_bolsa',0)>0)+int(d.get('invertido_cripto',0)>0)+int(d.get('invertido_propiedades',0)>0)
    d['patrimonio_liquido']     = d['dinero_banco']+d.get('invertido_bolsa',0)*0.95+d.get('invertido_cripto',0)*0.90
    d['ratio_affordability']    = d.get('presupuesto_max',d['patrimonio_neto']*0.8)/max(d['patrimonio_neto'],1)
    d['carga_familiar']         = d.get('dependientes_menores',0)/max(d.get('num_integrantes_familia',1),1)
    edad=d.get('edad',38)
    if edad<32: d['grupo_edad']='Joven'
    elif edad<45: d['grupo_edad']='Adulto_Medio'
    elif edad<58: d['grupo_edad']='Adulto_Mayor'
    else: d['grupo_edad']='Pre_Jubilacion'
    return d

def predict(data):
    defaults={'presupuesto_max':data.get('patrimonio_neto',0)*0.8,'invertido_bolsa':0,'invertido_cripto':0,'invertido_propiedades':0,'renta_pasiva':0,'gastos_mensuales':data.get('ingresos_anuales',0)*0.03,'anos_experiencia':8,'meses_empresa_actual':36,'num_integrantes_familia':3,'dependientes_menores':1,'genero':'Masculino','estado_civil':'Soltero','tipo_contrato':'Indefinido','motivo_compra':'Inversion','pais_origen':'India','zona_preferida':'Downtown Dubai','profesion_grupo':'Otras','sector_grupo':'Otros'}
    for k,v in defaults.items():
        if k not in data: data[k]=v
    data=feature_engineering(data)
    df=pd.DataFrame([data])
    for col in todas:
        if col not in df.columns: df[col]=0
    for col in log_features:
        if col in df.columns: df[col]=np.log1p(df[col].clip(lower=0))
    X=df[todas]
    prob=float(modelo.predict_proba(X)[0][1])
    pred=int(modelo.predict(X)[0])
    return prob,pred

def get_advice(patrimonio, deudas, ingresos, score):
    tips = []
    if patrimonio < 30000:
        needed = 30000 - patrimonio
        tips.append(f"💰 Incrementa patrimonio neto en <b>${needed:,.0f}</b> para superar el umbral mínimo")
    if (deudas/max(ingresos,1))*100 >= 40:
        max_deuda = ingresos * 0.40
        tips.append(f"📉 Reduce deudas a menos de <b>${max_deuda:,.0f}</b> (ratio actual {(deudas/max(ingresos,1))*100:.0f}%)")
    if score < 650:
        tips.append(f"⭐ Mejora tu score crediticio en <b>{650-score} puntos</b> (mínimo requerido: 650)")
    if not tips:
        tips.append("📊 Perfil muy cercano al umbral — pequeñas mejoras en deuda o patrimonio pueden cambiar el resultado")
    return tips[:3]

def build_gauge(pct):
    color = "#10B981" if pct >= 60 else "#D4AF37" if pct >= 35 else "#B91C1C"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 42, 'color': color, 'family': 'Cormorant Garamond'}},
        gauge={
            'axis': {'range': [0,100], 'tickwidth': 1, 'tickcolor': "#3a3530", 'tickfont': {'color': '#3a3530', 'size': 10}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#1A1A1A",
            'borderwidth': 0,
            'steps': [
                {'range': [0,35],  'color': 'rgba(185,28,28,0.15)'},
                {'range': [35,60], 'color': 'rgba(212,175,55,0.15)'},
                {'range': [60,100],'color': 'rgba(16,185,129,0.15)'},
            ],
            'threshold': {'line': {'color': color, 'width': 3}, 'thickness': 0.8, 'value': pct}
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=10, b=10), height=200,
        font={'family': 'Inter'}
    )
    return fig

def build_map(prob, presupuesto):
    m = folium.Map(location=[25.13,55.22], zoom_start=12, tiles='CartoDB dark_matter', attr='CartoDB')
    for z in ZONAS:
        match = min(100, max(0, int((presupuesto / z['min']) * 50 + prob * 50)))
        asequible = presupuesto >= z['min'] and prob >= 0.35
        perfecto  = presupuesto >= z['min'] and prob >= 0.60
        color  = '#10B981' if perfecto else '#D4AF37' if asequible else '#B91C1C'
        radius = max(400, min(1200, int(match * 10)))
        fit    = '✓ Zona ideal' if perfecto else '~ Asequible' if asequible else '✗ Fuera de rango'
        folium.Circle(
            location=[z['lat'],z['lng']], radius=radius, color=color,
            fill=True, fill_color=color, fill_opacity=0.2, weight=2, opacity=0.8,
            tooltip=folium.Tooltip(
                f"<div style='font-family:Inter,sans-serif;padding:6px;min-width:180px'>"
                f"<b style='font-size:13px'>{z['emoji']} {z['name']}</b><br/>"
                f"<span style='font-size:11px;color:#888'>{z['desc']}</span><br/><br/>"
                f"<span style='font-size:11px'>💶 Precio medio: <b>${z['m2']:,}/m²</b></span><br/>"
                f"<span style='font-size:11px'>🏷 Entrada desde: <b>${z['min']:,}</b></span><br/>"
                f"<span style='font-size:11px'>📊 Match score: <b>{match}%</b></span><br/>"
                f"<b style='color:{color};font-size:12px'>{fit}</b></div>",
                sticky=True
            )
        ).add_to(m)
        folium.CircleMarker(
            location=[z['lat'],z['lng']], radius=6, color=color,
            fill=True, fill_color=color, fill_opacity=1, weight=0
        ).add_to(m)
        folium.map.Marker(
            [z['lat']+0.006, z['lng']],
            icon=folium.DivIcon(
                html=f"<div style='font-family:Inter,sans-serif;font-size:10px;color:{color};font-weight:600;text-shadow:0 0 6px #000;white-space:nowrap'>{z['name']}</div>",
                icon_size=(160,20), icon_anchor=(80,0)
            )
        ).add_to(m)
    # Leyenda fija
    legend = """
    <div style='position:fixed;bottom:20px;left:20px;z-index:9999;
         background:rgba(14,14,14,0.92);border:1px solid rgba(212,175,55,0.3);
         border-radius:10px;padding:10px 14px;font-family:Inter,sans-serif;font-size:11px'>
      <div style='color:#D4AF37;font-weight:600;letter-spacing:2px;font-size:9px;text-transform:uppercase;margin-bottom:6px'>Leyenda</div>
      <div style='color:#10B981'>● Zona ideal</div>
      <div style='color:#D4AF37;margin:3px 0'>● Asequible</div>
      <div style='color:#B91C1C'>● Fuera de rango</div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE PARA PRESETS
# ═══════════════════════════════════════════════════════════════════
def init_state():
    if 'ingresos' not in st.session_state:
        st.session_state.update({'ingresos':120000,'patrimonio':80000,'deudas':30000,'banco':40000,'score':700,'edad':38,'cat':'Senior','interes':'Medio','edu':'Licenciatura'})

init_state()

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <div style="display:flex;align-items:center;gap:16px">
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="10" fill="rgba(212,175,55,0.1)" stroke="rgba(212,175,55,0.3)" stroke-width="1"/>
      <path d="M8 32 L8 18 L12 18 L12 12 L16 12 L16 8 L18 8 L18 12 L20 10 L22 12 L22 8 L24 8 L24 12 L28 12 L28 18 L32 18 L32 32 Z" fill="#D4AF37" opacity="0.85"/>
      <rect x="16" y="23" width="4" height="9" fill="#0E0E0E" opacity="0.9"/>
      <rect x="20" y="23" width="4" height="9" fill="#0E0E0E" opacity="0.9"/>
      <circle cx="20" cy="9" r="2" fill="#fff" opacity="0.9"/>
    </svg>
    <div>
      <div class="hdr-logo">DUBAI <b>PROPTECH</b></div>
      <div style="font-size:0.62rem;color:#6a6055;letter-spacing:2px;text-transform:uppercase">Predicción de vivienda ideal · Screening IA</div>
    </div>
  </div>
  <div class="hdr-pills">
    <span class="pill">AUC 0.9697</span>
    <span class="pill">F1-Score 0.8246</span>
    <span class="pill">10K clientes</span>
    <span class="pill">XGBoost · v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONTENEDOR CENTRAL
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div class='wrap'>", unsafe_allow_html=True)

# ── PRESETS ──────────────────────────────────────────────────────
st.markdown("<div class='sec-title'>Perfiles de referencia — selecciona para autocompletar</div>", unsafe_allow_html=True)

cols_p = st.columns(5)
preset_names = list(PRESETS.keys())
for i, col in enumerate(cols_p):
    p = PRESETS[preset_names[i]]
    with col:
        st.markdown(f"""
        <div class="preset-card">
          <div class="preset-icon">{p['icon']}</div>
          <div class="preset-name">{preset_names[i].split(' ',1)[1]}</div>
          <div class="preset-tag">{p['tag']}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Seleccionar", key=f"pre_{i}"):
            st.session_state.update({'ingresos':p['ingresos'],'patrimonio':p['patrimonio'],'deudas':p['deudas'],'banco':p['banco'],'score':p['score'],'edad':p['edad'],'cat':p['cat'],'interes':p['interes'],'edu':p['edu']})
            st.rerun()

# ── COLUMNAS PRINCIPALES ─────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")

with col_left:
    st.markdown("<div class='sec-title'>Perfil financiero del cliente</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)

    ingresos   = st.slider("Ingresos anuales (USD)",   20000, 1000000, st.session_state.ingresos,   5000,  format="$%d", key="sl_ing")
    patrimonio = st.slider("Patrimonio neto (USD)",         0,  5000000, st.session_state.patrimonio, 10000, format="$%d", key="sl_pat")
    deudas     = st.slider("Deudas totales (USD)",          0,   500000, st.session_state.deudas,     5000,  format="$%d", key="sl_deu")
    banco      = st.slider("Dinero en banco (USD)",         0,  1000000, st.session_state.banco,      5000,  format="$%d", key="sl_ban")
    score      = st.slider("Score crediticio",            300,      850, st.session_state.score,        10,               key="sl_scr")
    edad       = st.slider("Edad",                         22,       70, st.session_state.edad,          1,               key="sl_age")

    ca, cb = st.columns(2)
    with ca:
        cat_idx = ["Mid-Level","Senior","Director","C-Level"].index(st.session_state.cat)
        cat_prof = st.selectbox("Categoría profesional", ["Mid-Level","Senior","Director","C-Level"], index=cat_idx)
        edu_idx  = ["Bachillerato","Licenciatura","Master","PhD"].index(st.session_state.edu)
        nivel_edu = st.selectbox("Nivel educativo", ["Bachillerato","Licenciatura","Master","PhD"], index=edu_idx)
    with cb:
        int_idx  = ["Bajo","Medio","Alto"].index(st.session_state.interes)
        interes  = st.selectbox("Interés en compra", ["Bajo","Medio","Alto"], index=int_idx)

    st.markdown("</div>", unsafe_allow_html=True)

    # Actualizar session_state con valores actuales de sliders
    st.session_state.ingresos   = ingresos
    st.session_state.patrimonio = patrimonio
    st.session_state.deudas     = deudas
    st.session_state.banco      = banco
    st.session_state.score      = score
    st.session_state.edad       = edad
    st.session_state.cat        = cat_prof
    st.session_state.interes    = interes
    st.session_state.edu        = nivel_edu

# ── PREDICCIÓN EN TIEMPO REAL ────────────────────────────────────
with st.spinner("Analizando perfil en el modelo XGBoost…"):
    input_data = {
        'ingresos_anuales': ingresos, 'patrimonio_neto': patrimonio,
        'deudas_totales': deudas, 'dinero_banco': banco,
        'score_crediticio': score, 'edad': edad,
        'categoria_profesional': cat_prof, 'interes_compra_dubai': interes,
        'nivel_educativo': nivel_edu, 'presupuesto_max': patrimonio * 0.8,
    }
    prob, pred = predict(input_data)

pct    = round(prob * 100, 1)
ratio  = (deudas / max(ingresos,1)) * 100
c1     = patrimonio >= 30000
c2     = ratio < 40
c3     = score >= 650
presupuesto = patrimonio * 0.6 + ingresos * 0.3
color  = '#10B981' if pct >= 60 else '#D4AF37' if pct >= 35 else '#ef4444'
vtext  = 'Cliente viable' if pct >= 60 else 'Perfil borderline' if pct >= 35 else 'No viable'
vcls   = 'badge-ok' if pct >= 60 else 'badge-maybe' if pct >= 35 else 'badge-no'
c_ok   = lambda ok: ("#10B981" if ok else "#ef4444")
c_ico  = lambda ok: ("✓" if ok else "✗")

with col_right:
    st.markdown("<div class='sec-title'>Resultado del modelo</div>", unsafe_allow_html=True)

    r1, r2 = st.columns([1,1])
    with r1:
        st.markdown("<div class='panel' style='text-align:center;padding:16px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.58rem;letter-spacing:3px;color:#6a6055;text-transform:uppercase;margin-bottom:4px'>Probabilidad de viabilidad</div>", unsafe_allow_html=True)
        fig = build_gauge(pct)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<div style='text-align:center;margin-top:-10px'><span class='badge {vcls}'>{vtext}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.58rem;letter-spacing:2px;color:#D4AF37;text-transform:uppercase;margin-bottom:10px'>Criterios del modelo</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='crit-row'><span class='crit-label'>Patrimonio ≥ $30K</span><span style='color:{c_ok(c1)};font-weight:600'>${patrimonio:,.0f} {c_ico(c1)}</span></div>
        <div class='crit-row'><span class='crit-label'>Ratio deuda &lt; 40%</span><span style='color:{c_ok(c2)};font-weight:600'>{ratio:.1f}% {c_ico(c2)}</span></div>
        <div class='crit-row'><span class='crit-label'>Score ≥ 650</span><span style='color:{c_ok(c3)};font-weight:600'>{score} pts {c_ico(c3)}</span></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── ADVICE si no viable ──
        if pct < 60:
            tips = get_advice(patrimonio, deudas, ingresos, score)
            tips_html = "".join([f"<div class='advice-item'><span class='advice-icon'>→</span><span>{t}</span></div>" for t in tips])
            st.markdown(f"""
            <div class='advice'>
              <div class='advice-title'>Cómo llegar a viable</div>
              {tips_html}
            </div>""", unsafe_allow_html=True)

        # ── EXPLICABILIDAD ──
        with st.expander("📊 Explicabilidad del modelo — features más relevantes"):
            cap_pago = max(0, (ingresos*0.30/12)-(deudas/12))
            feats = [
                ("capacidad_pago_mensual", min(100, int(cap_pago/500))),
                ("deudas_totales",         min(100, max(0, 100-int(ratio*2)))),
                ("patrimonio_neto",        min(100, int(patrimonio/50000)*10)),
                ("score_crediticio",       min(100, int((score-300)/550*100))),
                ("ingresos_anuales",       min(100, int(ingresos/10000))),
            ]
            for name, val in feats:
                st.markdown(f"""
                <div class='feat-bar-wrap'>
                  <div class='feat-label'><span>{name}</span><span style='color:#D4AF37'>{val}%</span></div>
                  <div class='feat-bar-bg'><div class='feat-bar' style='width:{val}%'></div></div>
                </div>""", unsafe_allow_html=True)

    # ── MAPA ──
    st.markdown("<div class='sec-title' style='margin-top:16px'>Zonas de Dubai — hover para ver detalles</div>", unsafe_allow_html=True)
    m = build_map(prob, presupuesto)
    st_folium(m, width=None, height=420, returned_objects=[])

    # ── ZONE CARDS ──
    zonas_ok = [z for z in ZONAS if presupuesto >= z['min'] and pct >= 35]
    if zonas_ok:
        st.markdown(f"<div class='sec-title'>Zonas recomendadas ({len(zonas_ok)} disponibles)</div>", unsafe_allow_html=True)
        cards = "<div class='zones-grid'>"
        for z in sorted(zonas_ok, key=lambda x: x['min']):
            perfecto = presupuesto >= z['min'] and pct >= 60
            col_c = '#10B981' if perfecto else '#D4AF37'
            cards += f"""<div class='zone-card' style='border-left-color:{col_c}'>
              <div class='zone-name' style='color:{col_c}'>{z['emoji']} {z['name']}</div>
              <div class='zone-desc'>{z['desc']}</div>
              <div class='zone-price' style='color:{col_c}'>desde ${z['min']:,} · ${z['m2']:,}/m²</div>
            </div>"""
        cards += "</div>"
        st.markdown(cards, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("<div class='footer'>Modelo XGBoost · v1.0 · Dataset sintético 10K clientes Dubai · Proyecto académico</div>", unsafe_allow_html=True)
