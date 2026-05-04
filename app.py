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
.block-container { padding: 0 !important; background: #0a0a0f; max-width: 100% !important; }
.hdr { background:#0d0d14;border-bottom:1px solid #c9a84c33;padding:16px 32px;display:flex;align-items:center;justify-content:space-between; }
.logo { font-family:'Cormorant Garamond',serif;font-size:1.4rem;font-weight:300;letter-spacing:5px;color:#c9a84c;text-transform:uppercase; }
.logo b { color:#e8e0d0;font-weight:600; }
.logo-sub { font-family:'DM Sans',sans-serif;font-size:0.75rem;color:#6a6055;letter-spacing:1px;margin-left:12px; }
.tag { font-size:0.6rem;letter-spacing:2px;color:#c9a84c;border:1px solid #c9a84c44;padding:4px 10px;text-transform:uppercase; }
.subtitle { font-size:0.7rem;letter-spacing:1px;color:#6a6055;text-align:center;padding:10px 32px;background:#0d0d14;border-bottom:1px solid #c9a84c11; }
.stSlider>div>div>div>div { background:#c9a84c !important; }
.stSlider>div>div>div { background:#2a2a35 !important; }
.stSlider label { font-size:0.65rem !important;letter-spacing:2px !important;color:#8a8070 !important;text-transform:uppercase !important; }
.stSelectbox label { font-size:0.65rem !important;letter-spacing:2px !important;color:#8a8070 !important;text-transform:uppercase !important; }
.stSelectbox>div>div { background:#13131e !important;border:1px solid #c9a84c22 !important;border-radius:0 !important;color:#e8e0d0 !important; }
.stButton button { width:100%;background:#c9a84c !important;color:#0a0a0f !important;border:none !important;padding:14px !important;font-size:0.7rem !important;font-weight:600 !important;letter-spacing:3px !important;text-transform:uppercase !important;border-radius:0 !important; }
.section-label { font-size:0.6rem;letter-spacing:3px;color:#c9a84c;text-transform:uppercase;margin-bottom:16px;padding-bottom:8px;border-bottom:1px solid #c9a84c22; }
.result-box { background:#0d0d14;border:1px solid #c9a84c33;padding:20px;text-align:center;margin-bottom:12px; }
.prob-num { font-family:'Cormorant Garamond',serif;font-size:4.5rem;font-weight:300;line-height:1; }
.verdict { font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;padding:5px 14px;display:inline-block;margin-top:8px; }
.v-ok { background:#4caf8222;color:#4caf82;border:1px solid #4caf8244; }
.v-no { background:#e24b4a22;color:#e24b4a;border:1px solid #e24b4a44; }
.v-maybe { background:#c9a84c22;color:#c9a84c;border:1px solid #c9a84c44; }
.crit-box { background:#13131e;border:1px solid #c9a84c22;padding:14px 16px;margin-bottom:12px; }
.crit-row { display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid #ffffff08;font-size:0.78rem; }
.crit-row:last-child { border-bottom:none; }
.crit-name { color:#8a8070;font-size:0.62rem;letter-spacing:1px;text-transform:uppercase; }
.zones-grid { display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px; }
.zone-card { background:#13131e;border-left:3px solid;padding:10px 12px; }
.zone-card-name { font-size:0.75rem;font-weight:500;margin-bottom:2px; }
.zone-card-desc { font-size:0.62rem;color:#6a6055;letter-spacing:0.5px; }
.zone-card-price { font-size:0.65rem;margin-top:4px; }
footer { display:none !important; }
#MainMenu { display:none !important; }
header { display:none !important; }
section[data-testid="stSidebar"] { display:none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hdr">
  <div style="display:flex;align-items:center;gap:16px">
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
      <rect width="36" height="36" fill="#c9a84c" opacity="0.1"/>
      <path d="M6 28 L6 16 L10 16 L10 10 L14 10 L14 6 L16 6 L16 10 L18 8 L20 10 L20 6 L22 6 L22 10 L26 10 L26 16 L30 16 L30 28 Z" fill="#c9a84c" opacity="0.9"/>
      <rect x="14" y="20" width="4" height="8" fill="#0a0a0f" opacity="0.8"/>
      <rect x="18" y="20" width="4" height="8" fill="#0a0a0f" opacity="0.8"/>
      <circle cx="18" cy="8" r="1.5" fill="#e0bc60"/>
    </svg>
    <div>
      <div class="logo">DUBAI <b>PROPTECH</b></div>
      <div style="font-size:0.62rem;color:#6a6055;letter-spacing:2px;text-transform:uppercase;margin-top:1px">Predicción de vivienda ideal por perfil socioeconómico</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <div style="text-align:right">
      <div class="tag" style="margin-bottom:4px;display:inline-block">XGBoost Optimizado</div><br/>
      <span style="font-size:0.58rem;color:#6a6055;letter-spacing:1px">AUC 0.9697 · F1-Score 0.8246 · 10K clientes</span>
    </div>
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <circle cx="16" cy="16" r="15" stroke="#c9a84c" stroke-width="1" opacity="0.4"/>
      <path d="M16 4 L16 28 M4 16 L28 16" stroke="#c9a84c" stroke-width="0.5" opacity="0.3"/>
      <circle cx="16" cy="16" r="6" stroke="#c9a84c" stroke-width="1" opacity="0.6"/>
      <circle cx="16" cy="16" r="2" fill="#c9a84c"/>
      <path d="M16 10 L17.5 14 L16 13 L14.5 14 Z" fill="#c9a84c"/>
    </svg>
  </div>
</div>
<div class="subtitle">
  <span style="color:#c9a84c">◆</span>
  &nbsp;Introduce el perfil financiero del cliente · El modelo XGBoost predice su viabilidad de compra y recomienda zonas en Dubai&nbsp;
  <span style="color:#c9a84c">◆</span>
</div>
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
    {"id":"downtown","name":"Downtown Dubai","lat":25.1972,"lng":55.2744,"min":400000,"desc":"Burj Khalifa · Lujo absoluto","emoji":"🏙️"},
    {"id":"marina",  "name":"Dubai Marina",  "lat":25.0804,"lng":55.1404,"min":250000,"desc":"Frente al mar · Cosmopolita","emoji":"⛵"},
    {"id":"palm",    "name":"Palm Jumeirah", "lat":25.1124,"lng":55.1390,"min":800000,"desc":"Isla artificial · Ultra premium","emoji":"🌴"},
    {"id":"business","name":"Business Bay",  "lat":25.1862,"lng":55.2914,"min":300000,"desc":"Canal de Dubai · Negocios","emoji":"💼"},
    {"id":"creek",   "name":"Creek Harbour", "lat":25.2198,"lng":55.3524,"min":200000,"desc":"Nuevo desarrollo · Gran potencial","emoji":"🌊"},
    {"id":"jvc",     "name":"Jumeirah Village","lat":25.0549,"lng":55.2066,"min":120000,"desc":"Accesible · Comunidad familiar","emoji":"🏡"},
    {"id":"difc",    "name":"DIFC",          "lat":25.2124,"lng":55.2814,"min":350000,"desc":"Distrito financiero internacional","emoji":"🏦"},
    {"id":"jbr",     "name":"JBR Beach",     "lat":25.0759,"lng":55.1319,"min":280000,"desc":"The Walk · Primera línea de playa","emoji":"🏖️"},
]

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

def build_map(prob,presupuesto):
    m=folium.Map(location=[25.15,55.22],zoom_start=12,tiles='CartoDB dark_matter',attr='CartoDB')
    for z in ZONAS:
        asequible=presupuesto>=z['min'] and prob>=0.35
        perfecto=presupuesto>=z['min'] and prob>=0.60
        color='#4caf82' if perfecto else '#c9a84c' if asequible else '#e24b4a'
        fill=0.25 if perfecto else 0.18 if asequible else 0.10
        radius=900 if perfecto else 700 if asequible else 500
        fit='✓ Zona ideal' if perfecto else '~ Asequible' if asequible else '✗ Fuera de rango'
        folium.Circle(location=[z['lat'],z['lng']],radius=radius,color=color,fill=True,fill_color=color,fill_opacity=fill,weight=2,opacity=0.9,
            tooltip=folium.Tooltip(f"<div style='font-family:DM Sans,sans-serif;padding:4px'><b style='font-size:13px'>{z['emoji']} {z['name']}</b><br/><span style='font-size:11px;color:#666'>{z['desc']}</span><br/><span style='font-size:11px'>Precio mínimo: ${z['min']:,}</span><br/><b style='color:{color};font-size:12px'>{fit}</b></div>",sticky=True)
        ).add_to(m)
        folium.CircleMarker(location=[z['lat'],z['lng']],radius=5,color=color,fill=True,fill_color=color,fill_opacity=1).add_to(m)
        folium.map.Marker([z['lat']+0.006,z['lng']],icon=folium.DivIcon(html=f"<div style='font-family:DM Sans,sans-serif;font-size:10px;color:{color};font-weight:500;text-shadow:0 0 6px #000;white-space:nowrap'>{z['name']}</div>",icon_size=(150,20),icon_anchor=(75,0))).add_to(m)
    return m

col_form,col_results=st.columns([1,1.7],gap="small")

with col_form:
    st.markdown("<div style='background:#0d0d14;padding:24px 24px 8px 24px;border-right:1px solid #c9a84c22;min-height:85vh'>",unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Perfil financiero del cliente</div>",unsafe_allow_html=True)
    ingresos  =st.slider("Ingresos anuales (USD)",  20000,500000,120000,5000, format="$%d")
    patrimonio=st.slider("Patrimonio neto (USD)",       0,600000, 80000,5000, format="$%d")
    deudas    =st.slider("Deudas totales (USD)",        0,300000, 30000,5000, format="$%d")
    banco     =st.slider("Dinero en banco (USD)",       0,200000, 40000,5000, format="$%d")
    score     =st.slider("Score crediticio",          300,    850,   700,  10)
    edad      =st.slider("Edad",                       22,     70,    38,   1)
    ca,cb=st.columns(2)
    with ca:
        cat_prof =st.selectbox("Categoría profesional",["Mid-Level","Senior","Director","C-Level"],index=1)
        nivel_edu=st.selectbox("Nivel educativo",["Bachillerato","Licenciatura","Master","PhD"],index=1)
    with cb:
        interes=st.selectbox("Interés en compra",["Bajo","Medio","Alto"],index=1)
    st.markdown("</div>",unsafe_allow_html=True)

input_data={'ingresos_anuales':ingresos,'patrimonio_neto':patrimonio,'deudas_totales':deudas,'dinero_banco':banco,'score_crediticio':score,'edad':edad,'categoria_profesional':cat_prof,'interes_compra_dubai':interes,'nivel_educativo':nivel_edu,'presupuesto_max':patrimonio*0.8}
prob,pred=predict(input_data)
pct=round(prob*100,1)
ratio=(deudas/max(ingresos,1))*100
c1=patrimonio>=30000
c2=ratio<40
c3=score>=650
presupuesto=patrimonio*0.6+ingresos*0.3
color='#4caf82' if pct>=60 else '#c9a84c' if pct>=35 else '#e24b4a'
vtext='Cliente viable' if pct>=60 else 'Perfil borderline' if pct>=35 else 'No viable'
vcls='v-ok' if pct>=60 else 'v-maybe' if pct>=35 else 'v-no'
c_ok=lambda ok:("#4caf82" if ok else "#e24b4a")
c_ico=lambda ok:("✓" if ok else "✗")

with col_results:
    st.markdown("<div style='padding:20px 24px'>",unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-box">
      <div style="font-size:0.6rem;letter-spacing:3px;color:#6a6055;text-transform:uppercase;margin-bottom:6px">Probabilidad de viabilidad · XGBoost Optimizado</div>
      <div class="prob-num" style="color:{color}">{pct}%</div>
      <div><span class="verdict {vcls}">{vtext}</span></div>
    </div>""",unsafe_allow_html=True)
    st.markdown(f"""
    <div class="crit-box">
      <div style="font-size:0.6rem;letter-spacing:2px;color:#c9a84c;text-transform:uppercase;margin-bottom:8px">Criterios del modelo</div>
      <div class="crit-row"><span class="crit-name">Patrimonio neto ≥ $30,000</span><span style="color:{c_ok(c1)};font-weight:500">${patrimonio:,.0f} {c_ico(c1)}</span></div>
      <div class="crit-row"><span class="crit-name">Ratio deuda / ingreso &lt; 40%</span><span style="color:{c_ok(c2)};font-weight:500">{ratio:.1f}% {c_ico(c2)}</span></div>
      <div class="crit-row"><span class="crit-name">Score crediticio ≥ 650</span><span style="color:{c_ok(c3)};font-weight:500">{score} pts {c_ico(c3)}</span></div>
    </div>""",unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.6rem;letter-spacing:2px;color:#c9a84c;text-transform:uppercase;margin-bottom:6px'>Zonas de Dubai — pasa el cursor sobre cada zona</div>",unsafe_allow_html=True)
    m=build_map(prob,presupuesto)
    st_folium(m,width=None,height=360,returned_objects=[])
    st.markdown("""<div style="display:flex;gap:20px;margin:6px 0 14px 0;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;color:#6a6055">
      <span style="color:#4caf82">● Zona ideal</span><span style="color:#c9a84c">● Asequible</span><span style="color:#e24b4a">● Fuera de rango</span></div>""",unsafe_allow_html=True)
    zonas_ok=[z for z in ZONAS if presupuesto>=z['min'] and pct>=35]
    if zonas_ok:
        st.markdown(f"<div style='font-size:0.6rem;letter-spacing:2px;color:#c9a84c;text-transform:uppercase;margin-bottom:8px'>Zonas recomendadas ({len(zonas_ok)} encontradas)</div>",unsafe_allow_html=True)
        cards_html="<div class='zones-grid'>"
        for z in sorted(zonas_ok,key=lambda x:x['min']):
            perfecto=presupuesto>=z['min'] and pct>=60
            col_card='#4caf82' if perfecto else '#c9a84c'
            cards_html+=f"<div class='zone-card' style='border-left-color:{col_card}'><div class='zone-card-name' style='color:{col_card}'>{z['emoji']} {z['name']}</div><div class='zone-card-desc'>{z['desc']}</div><div class='zone-card-price' style='color:{col_card}'>desde ${z['min']:,}</div></div>"
        cards_html+="</div>"
        st.markdown(cards_html,unsafe_allow_html=True)
    else:
        st.markdown("<div style='background:#13131e;border:1px solid #e24b4a22;padding:12px 16px;font-size:0.75rem;color:#8a8070'>Ninguna zona es asequible con el perfil actual. Mejora el patrimonio neto o reduce las deudas.</div>",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
