import streamlit as st
import joblib, json, numpy as np, pandas as pd
import folium, plotly.graph_objects as go
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dubai PropTech · AI Screening", page_icon="🏙️", layout="wide", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.main, [data-testid="stAppViewContainer"] { background: #080810 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; background: #080810 !important; }
section[data-testid="stSidebar"], footer, header { display: none !important; }
#MainMenu { display: none !important; }

/* ── HEADER ── */
.hdr {
  background: linear-gradient(180deg, #0d0d1a 0%, #080810 100%);
  border-bottom: 1px solid rgba(212,175,55,0.2);
  padding: 0 40px;
  height: 64px;
  display: flex; align-items: center; justify-content: space-between;
}
.hdr-left { display:flex; align-items:center; gap:14px; }
.hdr-logo-text { line-height:1; }
.hdr-logo-main { font-family:'Cormorant Garamond',serif; font-size:1.5rem; font-weight:300; letter-spacing:8px; color:#D4AF37; text-transform:uppercase; }
.hdr-logo-main b { color:#fff; font-weight:600; }
.hdr-logo-sub { font-size:0.58rem; letter-spacing:3px; color:#4a4560; text-transform:uppercase; margin-top:2px; }
.hdr-pills { display:flex; gap:8px; }
.pill {
  font-size:0.58rem; letter-spacing:1.5px; font-weight:600; text-transform:uppercase;
  padding:5px 12px; border-radius:20px; color:#D4AF37;
  border:1px solid rgba(212,175,55,0.35); background:rgba(212,175,55,0.07);
  animation: glow 3s ease-in-out infinite;
}
@keyframes glow { 0%,100%{box-shadow:0 0 0 0 rgba(212,175,55,0.2)} 50%{box-shadow:0 0 8px 2px rgba(212,175,55,0.1)} }

/* ── WRAP ── */
.wrap { max-width:1440px; margin:0 auto; padding:24px 32px 48px; }

/* ── SECTION LABELS ── */
.sec-label {
  font-size:0.58rem; letter-spacing:4px; color:#D4AF37; text-transform:uppercase;
  font-weight:600; margin:0 0 14px 0; padding-bottom:10px;
  border-bottom:1px solid rgba(212,175,55,0.15);
  display:flex; align-items:center; gap:8px;
}
.sec-label::before { content:''; display:block; width:3px; height:14px; background:#D4AF37; border-radius:2px; }

/* ── PRESET CARDS ── */
.presets-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin-bottom:28px; }
.preset-card {
  background:#0f0f1c; border:1px solid rgba(212,175,55,0.15); border-radius:12px;
  padding:16px 14px 12px; text-align:center; transition:all 0.25s; cursor:pointer;
}
.preset-card:hover { border-color:rgba(212,175,55,0.6); background:#15152a; box-shadow:0 0 20px rgba(212,175,55,0.1); transform:translateY(-2px); }
.preset-icon-wrap { width:44px; height:44px; margin:0 auto 10px; border-radius:10px; display:flex; align-items:center; justify-content:center; }
.preset-name { font-size:0.75rem; font-weight:600; color:#e8e0d0; margin-bottom:3px; letter-spacing:0.3px; }
.preset-tag { font-size:0.58rem; color:#5a5570; letter-spacing:0.3px; line-height:1.4; }

/* ── PANEL ── */
.panel { background:#0f0f1c; border:1px solid rgba(212,175,55,0.12); border-radius:14px; padding:20px 18px; margin-bottom:14px; }

/* ── SLIDERS DORADOS ── */
.stSlider label { font-size:0.6rem !important; letter-spacing:2px !important; color:#5a5570 !important; text-transform:uppercase !important; }
.stSlider > div > div > div { background:#1a1a2e !important; border-radius:4px !important; height:4px !important; }
.stSlider > div > div > div > div { background:linear-gradient(90deg,#b8941f,#D4AF37,#e8cc60) !important; border-radius:4px !important; }
[data-testid="stSliderThumb"] { background:#D4AF37 !important; border:2px solid #080810 !important; box-shadow:0 0 12px rgba(212,175,55,0.7) !important; width:20px !important; height:20px !important; }

/* ── SELECTBOX ── */
.stSelectbox label { font-size:0.6rem !important; letter-spacing:2px !important; color:#5a5570 !important; text-transform:uppercase !important; }
.stSelectbox > div > div { background:#0a0a18 !important; border:1px solid rgba(212,175,55,0.18) !important; border-radius:8px !important; color:#e8e0d0 !important; }

/* ── BOTÓN ── */
.stButton > button {
  background:linear-gradient(135deg,#c9a020,#D4AF37,#c9a020) !important;
  color:#080810 !important; border:none !important; border-radius:8px !important;
  font-weight:700 !important; letter-spacing:2px !important; font-size:0.62rem !important;
  text-transform:uppercase !important; padding:10px !important; width:100% !important;
  transition:all 0.2s !important; box-shadow:0 2px 12px rgba(212,175,55,0.2) !important;
}
.stButton > button:hover { box-shadow:0 4px 20px rgba(212,175,55,0.4) !important; transform:translateY(-1px) !important; }

/* ── BADGES ── */
.badge { font-size:0.62rem; letter-spacing:2px; text-transform:uppercase; padding:6px 18px; border-radius:20px; display:inline-block; font-weight:700; margin-top:8px; }
.badge-ok    { background:rgba(16,185,129,0.12); color:#10B981; border:1px solid rgba(16,185,129,0.35); }
.badge-no    { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.35); }
.badge-maybe { background:rgba(212,175,55,0.12); color:#D4AF37; border:1px solid rgba(212,175,55,0.35); }

/* ── CRIT ROWS ── */
.crit-row { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.crit-row:last-child { border-bottom:none; }
.crit-label { font-size:0.6rem; letter-spacing:1.5px; text-transform:uppercase; color:#5a5570; }
.crit-val { font-size:0.82rem; font-weight:600; }

/* ── ADVICE ── */
.advice { background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.25); border-radius:10px; padding:14px 16px; margin-top:12px; }
.advice-title { font-size:0.58rem; letter-spacing:2px; color:#ef4444; text-transform:uppercase; font-weight:700; margin-bottom:10px; }
.advice-row { display:flex; gap:8px; margin-bottom:7px; font-size:0.78rem; color:#b0a090; align-items:flex-start; }
.advice-arrow { color:#D4AF37; font-weight:700; margin-top:1px; flex-shrink:0; }

/* ── FEAT BARS ── */
.feat-wrap { margin:8px 0; }
.feat-top { display:flex; justify-content:space-between; font-size:0.6rem; color:#5a5570; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.feat-bg { background:#1a1a2e; border-radius:4px; height:5px; }
.feat-fill { height:5px; border-radius:4px; background:linear-gradient(90deg,#b8941f,#D4AF37); }

/* ── ZONE CARDS ── */
.zone-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px; }
.zone-card { background:#0a0a18; border-radius:10px; border-left:3px solid; padding:12px 14px; display:flex; align-items:flex-start; gap:10px; }
.zone-card-icon { width:32px; height:32px; flex-shrink:0; border-radius:7px; display:flex; align-items:center; justify-content:center; }
.zone-card-body { min-width:0; }
.zone-card-name { font-size:0.75rem; font-weight:600; margin-bottom:2px; }
.zone-card-desc { font-size:0.6rem; color:#5a5570; margin-bottom:4px; }
.zone-card-price { font-size:0.65rem; font-weight:500; }

/* ── FOOTER ── */
.footer { text-align:center; padding:24px; font-size:0.58rem; color:#2a2540; letter-spacing:3px; text-transform:uppercase; border-top:1px solid rgba(212,175,55,0.08); margin-top:16px; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SVG ICONS
# ══════════════════════════════════════════════════════════════════════
ICONS = {
    "whale":   ('<svg viewBox="0 0 24 24" fill="none" stroke="#D4AF37" stroke-width="1.5" width="22" height="22"><path d="M12 2L8 8H4l4 4-2 6 6-3 6 3-2-6 4-4h-4L12 2z"/></svg>', "#1a1500", "#D4AF37"),
    "expat":   ('<svg viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="1.5" width="22" height="22"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2"/><line x1="12" y1="12" x2="12" y2="16"/><line x1="10" y1="14" x2="14" y2="14"/></svg>', "#0a1525", "#60a5fa"),
    "tech":    ('<svg viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.5" width="22" height="22"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>', "#120a25", "#a78bfa"),
    "familia": ('<svg viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="1.5" width="22" height="22"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>', "#0a1a12", "#34d399"),
    "joven":   ('<svg viewBox="0 0 24 24" fill="none" stroke="#fb923c" stroke-width="1.5" width="22" height="22"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>', "#1a0a00", "#fb923c"),
}

ZONE_ICONS = {
    "downtown": '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><rect x="2" y="3" width="20" height="18" rx="1"/><line x1="9" y1="21" x2="9" y2="9"/><line x1="15" y1="21" x2="15" y2="9"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "marina":   '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><path d="M2 20h20M5 20V10l7-7 7 7v10"/><path d="M9 20v-5h6v5"/></svg>',
    "palm":     '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><path d="M12 22V12M12 12C12 12 7 8 4 5c4 0 6 2 8 4M12 12c0 0 5-4 8-7-4 0-6 2-8 4"/><path d="M8 22h8"/></svg>',
    "business": '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>',
    "creek":    '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><circle cx="12" cy="12" r="10"/><path d="M12 2v20M2 12h20"/></svg>',
    "jvc":      '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "difc":     '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6"/></svg>',
    "jbr":      '<svg viewBox="0 0 24 24" fill="none" stroke="COLOR" stroke-width="1.5" width="18" height="18"><path d="M2 18h20M6 18V9.5L12 4l6 5.5V18"/><circle cx="12" cy="11" r="2"/></svg>',
}

# ══════════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════════
PRESETS = [
    {"key":"whale",   "name":"Whale Investor",     "tag":"C-Level · Zonas ultra premium",         "icon":"whale",  "ingresos":850000,"patrimonio":4500000,"deudas":200000,"banco":900000,"score":820,"edad":55,"cat":"C-Level","interes":"Alto","edu":"PhD"},
    {"key":"expat",   "name":"Expat Ejecutivo",     "tag":"Senior Manager · Múltiples zonas",      "icon":"expat",  "ingresos":280000,"patrimonio":650000, "deudas":120000,"banco":180000,"score":760,"edad":42,"cat":"Director","interes":"Alto","edu":"Master"},
    {"key":"tech",    "name":"Tech Founder",         "tag":"Emprendedor · Alto potencial",          "icon":"tech",   "ingresos":180000,"patrimonio":400000, "deudas":80000, "banco":120000,"score":720,"edad":34,"cat":"Senior","interes":"Alto","edu":"Master"},
    {"key":"familia", "name":"Familia Profesional",  "tag":"Mid-Senior · Zonas residenciales",      "icon":"familia","ingresos":140000,"patrimonio":220000, "deudas":90000, "banco":60000, "score":700,"edad":38,"cat":"Senior","interes":"Medio","edu":"Licenciatura"},
    {"key":"joven",   "name":"Joven Profesional",    "tag":"Junior · Perfil borderline",            "icon":"joven",  "ingresos":75000, "patrimonio":40000,  "deudas":28000, "banco":12000, "score":672,"edad":27,"cat":"Mid-Level","interes":"Medio","edu":"Licenciatura"},
]

# ══════════════════════════════════════════════════════════════════════
# ZONAS
# ══════════════════════════════════════════════════════════════════════
ZONAS = [
    {"id":"downtown","name":"Downtown Dubai", "lat":25.1972,"lng":55.2744,"min":350000,"m2":22000,"desc":"Burj Khalifa · Lujo absoluto"},
    {"id":"marina",  "name":"Dubai Marina",   "lat":25.0804,"lng":55.1404,"min":200000,"m2":16000,"desc":"Frente al mar · Cosmopolita"},
    {"id":"palm",    "name":"Palm Jumeirah",  "lat":25.1124,"lng":55.1390,"min":600000,"m2":35000,"desc":"Isla artificial · Ultra premium"},
    {"id":"business","name":"Business Bay",   "lat":25.1862,"lng":55.2914,"min":250000,"m2":18000,"desc":"Canal de Dubai · Negocios"},
    {"id":"creek",   "name":"Creek Harbour",  "lat":25.2198,"lng":55.3524,"min":150000,"m2":14000,"desc":"Nuevo desarrollo · Gran potencial"},
    {"id":"jvc",     "name":"Jumeirah Village","lat":25.0549,"lng":55.2066,"min":80000, "m2":9000, "desc":"Accesible · Comunidad familiar"},
    {"id":"difc",    "name":"DIFC",           "lat":25.2124,"lng":55.2814,"min":300000,"m2":20000,"desc":"Distrito financiero internacional"},
    {"id":"jbr",     "name":"JBR Beach",      "lat":25.0759,"lng":55.1319,"min":220000,"m2":17000,"desc":"The Walk · Primera línea playa"},
]

# ══════════════════════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════════════════════
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
    tips=[]
    if patrimonio<30000: tips.append(f"Incrementa patrimonio neto en <b>${30000-patrimonio:,.0f}</b> para superar el umbral mínimo de $30,000")
    if (deudas/max(ingresos,1))*100>=40: tips.append(f"Reduce deudas a menos de <b>${ingresos*0.40:,.0f}</b> (ratio actual {(deudas/max(ingresos,1))*100:.0f}%&nbsp;→ máximo 40%)")
    if score<650: tips.append(f"Mejora tu score crediticio en <b>{650-score} puntos</b> — actualmente {score}, mínimo requerido 650")
    if not tips: tips.append("Perfil muy cercano al umbral — pequeñas mejoras en deuda o patrimonio pueden cambiar el resultado")
    return tips[:3]

def build_gauge(pct):
    color="#10B981" if pct>=60 else "#D4AF37" if pct>=35 else "#ef4444"
    fig=go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix':'%','font':{'size':46,'color':color,'family':'Cormorant Garamond'}},
        gauge={
            'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#2a2540','tickfont':{'color':'#2a2540','size':9}},
            'bar':{'color':color,'thickness':0.28},
            'bgcolor':'#0f0f1c','borderwidth':0,
            'steps':[
                {'range':[0,35],'color':'rgba(239,68,68,0.12)'},
                {'range':[35,60],'color':'rgba(212,175,55,0.12)'},
                {'range':[60,100],'color':'rgba(16,185,129,0.12)'},
            ],
            'threshold':{'line':{'color':color,'width':3},'thickness':0.85,'value':pct}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=16,r=16,t=8,b=8),height=190,font={'family':'Inter'})
    return fig

def build_map(prob, presupuesto):
    m=folium.Map(location=[25.13,55.22],zoom_start=11,tiles='CartoDB dark_matter',attr='CartoDB',
                 zoom_control=False,scrollWheelZoom=False,dragging=False,doubleClickZoom=False,touchZoom=False)
    for z in ZONAS:
        match=min(100,max(0,int((presupuesto/z['min'])*50+prob*50)))
        asequible=presupuesto>=z['min'] and prob>=0.35
        perfecto=presupuesto>=z['min'] and prob>=0.60
        color='#10B981' if perfecto else '#D4AF37' if asequible else '#ef4444'
        glow='rgba(16,185,129,0.3)' if perfecto else 'rgba(212,175,55,0.3)' if asequible else 'rgba(239,68,68,0.2)'
        radius=max(350,min(900,int(match*8)))
        fit='✓ Zona ideal' if perfecto else '~ Asequible' if asequible else '✗ Fuera de rango'
        tooltip_html=(
            f"<div style='font-family:Inter,sans-serif;padding:10px 12px;min-width:200px;"
            f"background:#0f0f1c;border:1px solid {color};border-radius:10px;box-shadow:0 4px 20px rgba(0,0,0,0.5)'>"
            f"<div style='font-size:13px;font-weight:700;color:{color};margin-bottom:4px'>{z['name']}</div>"
            f"<div style='font-size:11px;color:#6a6580;margin-bottom:8px'>{z['desc']}</div>"
            f"<div style='display:flex;gap:16px;margin-bottom:6px'>"
            f"<div><div style='font-size:9px;color:#4a4560;text-transform:uppercase;letter-spacing:1px'>Desde</div>"
            f"<div style='font-size:12px;color:#e8e0d0;font-weight:600'>${z['min']:,}</div></div>"
            f"<div><div style='font-size:9px;color:#4a4560;text-transform:uppercase;letter-spacing:1px'>Precio/m²</div>"
            f"<div style='font-size:12px;color:#e8e0d0;font-weight:600'>${z['m2']:,}</div></div>"
            f"<div><div style='font-size:9px;color:#4a4560;text-transform:uppercase;letter-spacing:1px'>Match</div>"
            f"<div style='font-size:12px;color:{color};font-weight:700'>{match}%</div></div>"
            f"</div>"
            f"<div style='font-size:11px;font-weight:700;color:{color}'>{fit}</div>"
            f"</div>"
        )
        folium.Circle(location=[z['lat'],z['lng']],radius=radius,color=color,fill=True,fill_color=color,fill_opacity=0.15,weight=1.5,opacity=0.7).add_to(m)
        folium.CircleMarker(location=[z['lat'],z['lng']],radius=9,color='#080810',fill=True,fill_color=color,fill_opacity=1,weight=2,
            tooltip=folium.Tooltip(tooltip_html,sticky=False,direction='top',offset=(0,-14))).add_to(m)
        folium.map.Marker([z['lat']+0.007,z['lng']],
            icon=folium.DivIcon(html=f"<div style='font-family:Inter,sans-serif;font-size:10px;font-weight:600;color:{color};text-shadow:0 0 6px #000,0 0 3px #000;white-space:nowrap;letter-spacing:0.5px'>{z['name']}</div>",
            icon_size=(160,18),icon_anchor=(80,0))).add_to(m)
    legend=f"""<div style='position:fixed;bottom:16px;left:16px;z-index:9999;background:rgba(8,8,16,0.95);
         border:1px solid rgba(212,175,55,0.25);border-radius:10px;padding:12px 16px;
         font-family:Inter,sans-serif'><div style='font-size:9px;letter-spacing:2px;color:#D4AF37;
         text-transform:uppercase;font-weight:700;margin-bottom:8px'>Leyenda</div>
         <div style='font-size:11px;color:#10B981;margin-bottom:4px'>● Zona ideal</div>
         <div style='font-size:11px;color:#D4AF37;margin-bottom:4px'>● Asequible</div>
         <div style='font-size:11px;color:#ef4444'>● Fuera de rango</div></div>"""
    m.get_root().html.add_child(folium.Element(legend))
    return m

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
def init():
    if 'ingresos' not in st.session_state:
        st.session_state.update({'ingresos':140000,'patrimonio':220000,'deudas':90000,'banco':60000,'score':700,'edad':38,'cat':'Senior','interes':'Medio','edu':'Licenciatura'})
init()

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hdr">
  <div class="hdr-left">
    <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="10" fill="rgba(212,175,55,0.08)" stroke="rgba(212,175,55,0.25)" stroke-width="1"/>
      <path d="M8 33L8 19L12 19L12 13L16 13L16 9L18 9L18 13L20 11L22 13L22 9L24 9L24 13L28 13L28 19L32 19L32 33Z" fill="#D4AF37" opacity="0.9"/>
      <rect x="15" y="24" width="4" height="9" fill="#080810"/>
      <rect x="21" y="24" width="4" height="9" fill="#080810"/>
      <circle cx="20" cy="9.5" r="2" fill="#fff" opacity="0.95"/>
    </svg>
    <div class="hdr-logo-text">
      <div class="hdr-logo-main">DUBAI <b>PROPTECH</b></div>
      <div class="hdr-logo-sub">Predicción de vivienda ideal · Screening IA</div>
    </div>
  </div>
  <div class="hdr-pills">
    <span class="pill">AUC 0.9697</span>
    <span class="pill">F1 0.8246</span>
    <span class="pill">10K clientes</span>
    <span class="pill">XGBoost · v1.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# CONTENIDO
# ══════════════════════════════════════════════════════════════════════
st.markdown("<div class='wrap'>", unsafe_allow_html=True)

# ── PRESETS ──
st.markdown("<div class='sec-label'>Perfiles de referencia — selecciona para autocompletar</div>", unsafe_allow_html=True)
st.markdown("<div class='presets-grid'>", unsafe_allow_html=True)
cols_p = st.columns(5)
for i, col in enumerate(cols_p):
    p = PRESETS[i]
    svg, bg, accent = ICONS[p["icon"]]
    with col:
        st.markdown(f"""
        <div class="preset-card">
          <div class="preset-icon-wrap" style="background:{bg};border:1px solid {accent}33">{svg}</div>
          <div class="preset-name">{p['name']}</div>
          <div class="preset-tag">{p['tag']}</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Seleccionar", key=f"pre_{i}"):
            st.session_state.update({'ingresos':p['ingresos'],'patrimonio':p['patrimonio'],'deudas':p['deudas'],'banco':p['banco'],'score':p['score'],'edad':p['edad'],'cat':p['cat'],'interes':p['interes'],'edu':p['edu']})
            st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ── COLUMNAS ──
col_l, col_r = st.columns([1, 1.85], gap="large")

with col_l:
    st.markdown("<div class='sec-label'>Perfil financiero del cliente</div>", unsafe_allow_html=True)
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    ingresos   = st.slider("Ingresos anuales (USD)",   20000, 1000000, st.session_state.ingresos,   5000,  format="$%d")
    patrimonio = st.slider("Patrimonio neto (USD)",         0,  5000000, st.session_state.patrimonio, 10000, format="$%d")
    deudas     = st.slider("Deudas totales (USD)",          0,   500000, st.session_state.deudas,     5000,  format="$%d")
    banco      = st.slider("Dinero en banco (USD)",         0,  1000000, st.session_state.banco,      5000,  format="$%d")
    score      = st.slider("Score crediticio",            300,      850, st.session_state.score,        10)
    edad       = st.slider("Edad",                         22,       70, st.session_state.edad,          1)
    ca, cb = st.columns(2)
    with ca:
        cat_idx = ["Mid-Level","Senior","Director","C-Level"].index(st.session_state.cat)
        cat_prof  = st.selectbox("Categoría profesional", ["Mid-Level","Senior","Director","C-Level"], index=cat_idx)
        edu_idx  = ["Bachillerato","Licenciatura","Master","PhD"].index(st.session_state.edu)
        nivel_edu = st.selectbox("Nivel educativo", ["Bachillerato","Licenciatura","Master","PhD"], index=edu_idx)
    with cb:
        int_idx = ["Bajo","Medio","Alto"].index(st.session_state.interes)
        interes = st.selectbox("Interés en compra", ["Bajo","Medio","Alto"], index=int_idx)
    st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.update({'ingresos':ingresos,'patrimonio':patrimonio,'deudas':deudas,'banco':banco,'score':score,'edad':edad,'cat':cat_prof,'interes':interes,'edu':nivel_edu})

# ── PREDICCIÓN ──
with st.spinner("Analizando perfil en el modelo XGBoost…"):
    prob, pred = predict({'ingresos_anuales':ingresos,'patrimonio_neto':patrimonio,'deudas_totales':deudas,'dinero_banco':banco,'score_crediticio':score,'edad':edad,'categoria_profesional':cat_prof,'interes_compra_dubai':interes,'nivel_educativo':nivel_edu,'presupuesto_max':patrimonio*0.8})

pct   = round(prob*100,1)
ratio = (deudas/max(ingresos,1))*100
c1,c2,c3 = patrimonio>=30000, ratio<40, score>=650
presupuesto = patrimonio*0.8 + ingresos*2.5
color = '#10B981' if pct>=60 else '#D4AF37' if pct>=35 else '#ef4444'
vtext = 'Cliente viable' if pct>=60 else 'Perfil borderline' if pct>=35 else 'No viable'
vcls  = 'badge-ok' if pct>=60 else 'badge-maybe' if pct>=35 else 'badge-no'
c_ok  = lambda ok: ("#10B981" if ok else "#ef4444")
c_ico = lambda ok: ("✓" if ok else "✗")

with col_r:
    st.markdown("<div class='sec-label'>Resultado del modelo</div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1,1])

    with r1:
        st.markdown("<div class='panel' style='text-align:center'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.58rem;letter-spacing:3px;color:#4a4560;text-transform:uppercase;margin-bottom:4px'>Probabilidad de viabilidad</div>", unsafe_allow_html=True)
        st.plotly_chart(build_gauge(pct), use_container_width=True, config={'displayModeBar':False})
        st.markdown(f"<div style='text-align:center;margin-top:-8px'><span class='badge {vcls}'>{vtext}</span></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with r2:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.58rem;letter-spacing:2px;color:#D4AF37;text-transform:uppercase;margin-bottom:12px;font-weight:700'>Criterios del modelo</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='crit-row'><span class='crit-label'>Patrimonio ≥ $30K</span><span class='crit-val' style='color:{c_ok(c1)}'>${patrimonio:,.0f} {c_ico(c1)}</span></div>
        <div class='crit-row'><span class='crit-label'>Ratio deuda &lt; 40%</span><span class='crit-val' style='color:{c_ok(c2)}'>{ratio:.1f}% {c_ico(c2)}</span></div>
        <div class='crit-row'><span class='crit-label'>Score ≥ 650</span><span class='crit-val' style='color:{c_ok(c3)}'>{score} pts {c_ico(c3)}</span></div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if pct < 60:
            tips = get_advice(patrimonio, deudas, ingresos, score)
            rows = "".join([f"<div class='advice-row'><span class='advice-arrow'>→</span><span>{t}</span></div>" for t in tips])
            st.markdown(f"<div class='advice'><div class='advice-title'>Cómo llegar a viable</div>{rows}</div>", unsafe_allow_html=True)
        with st.expander("📊 Explicabilidad del modelo"):
            cap_pago=max(0,(ingresos*0.30/12)-(deudas/12))
            feats=[("capacidad_pago_mensual",min(100,int(cap_pago/500))),("deudas_totales",min(100,max(0,100-int(ratio*2)))),("patrimonio_neto",min(100,int(patrimonio/50000)*10)),("score_crediticio",min(100,int((score-300)/550*100))),("ingresos_anuales",min(100,int(ingresos/10000)))]
            for name,val in feats:
                st.markdown(f"<div class='feat-wrap'><div class='feat-top'><span>{name}</span><span style='color:#D4AF37'>{val}%</span></div><div class='feat-bg'><div class='feat-fill' style='width:{val}%'></div></div></div>",unsafe_allow_html=True)

    # ── MAPA ──
    st.markdown("<div class='sec-label' style='margin-top:16px'>Zonas de Dubai — hover sobre cada punto para ver detalles</div>", unsafe_allow_html=True)
    st_folium(build_map(prob, presupuesto), width=None, height=400, returned_objects=[])

    # ── ZONE CARDS ──
    zonas_ok = [z for z in ZONAS if presupuesto>=z['min'] and pct>=35]
    if zonas_ok:
        st.markdown(f"<div class='sec-label' style='margin-top:16px'>Zonas recomendadas — {len(zonas_ok)} dentro de presupuesto</div>", unsafe_allow_html=True)
        html = "<div class='zone-grid'>"
        for z in sorted(zonas_ok, key=lambda x: x['min']):
            perfecto = presupuesto>=z['min'] and pct>=60
            col_c = '#10B981' if perfecto else '#D4AF37'
            bg_c  = 'rgba(16,185,129,0.08)' if perfecto else 'rgba(212,175,55,0.08)'
            svg_icon = ZONE_ICONS.get(z['id'],'').replace('COLOR', col_c)
            html += f"""<div class='zone-card' style='border-left-color:{col_c}'>
              <div class='zone-card-icon' style='background:{bg_c}'>{svg_icon}</div>
              <div class='zone-card-body'>
                <div class='zone-card-name' style='color:{col_c}'>{z['name']}</div>
                <div class='zone-card-desc'>{z['desc']}</div>
                <div class='zone-card-price' style='color:{col_c}'>desde ${z['min']:,} · ${z['m2']:,}/m²</div>
              </div>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>Modelo XGBoost Optimizado · v1.0 · Dataset sintético 10K clientes · Proyecto académico · Dubai PropTech</div>", unsafe_allow_html=True)
