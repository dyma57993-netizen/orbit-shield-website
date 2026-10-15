import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="ORBIT-SHIELD | Intelligent Orbital Simulation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter','IBM Plex Sans Arabic',sans-serif; }
.stApp { background: radial-gradient(ellipse at 75% 0%, rgba(17,75,120,.22), transparent 38%), linear-gradient(180deg,#07101d 0%,#030711 100%); color:#eaf2ff; }
.block-container { padding-top: 1.3rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1728,#07101c); border-right:1px solid #1c3048; }
div[data-testid="stMetric"] { background:rgba(13,29,49,.75); border:1px solid #203c58; padding:14px 16px; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:#8faac7; }
div[data-testid="stMetricValue"] { color:#f4f8ff; }
div.stButton > button { border-radius:10px; border:1px solid #2a83b8; background:#0b6e9f; color:white; font-weight:700; }
.small-muted { color:#91a9c4; font-size:.9rem; }
.hero { padding: 22px 24px; border:1px solid #203c58; border-radius:18px; background:linear-gradient(110deg,rgba(11,36,62,.96),rgba(6,15,29,.88)); margin-bottom:16px; }
.pill { display:inline-block; border:1px solid #27617d; border-radius:99px; padding:4px 10px; color:#76d8ff; font-size:12px; letter-spacing:.05em; }
.note { border-left:3px solid #45b8e8; background:rgba(16,45,70,.48); padding:12px 14px; border-radius:8px; color:#c7d9ec; }
</style>
""", unsafe_allow_html=True)

EARTH_R = 6371.0
MU = 398600.4418  # km^3/s^2
SAT_ALT_DEFAULT = 500.0
MANEUVERS = [-200, -100, 0, 100, 200]
HORIZON_MIN = 90
N_TIME = 91

def orbital_rate(alt_km):
    """Mean motion for a circular orbit, rad/s; simplified two-body assumption."""
    radius = EARTH_R + float(alt_km)
    return np.sqrt(MU / radius**3)

def closest_approach(sat_alt, debris_alt, phase_deg, uncertainty_deg=0.0,
                     horizon_min=HORIZON_MIN, n_time=N_TIME):
    """Minimum separation over a short, simplified, coplanar circular-orbit window."""
    times = np.linspace(0, horizon_min * 60, n_time)
    rs = EARTH_R + sat_alt
    rd = EARTH_R + debris_alt
    ns = orbital_rate(sat_alt)
    nd = orbital_rate(debris_alt)
    results = []
    for offset in (-uncertainty_deg, 0.0, uncertainty_deg):
        phase0 = np.radians(phase_deg + offset)
        phase = phase0 + (nd - ns) * times
        d = np.sqrt(rs**2 + rd**2 - 2 * rs * rd * np.cos(phase))
        results.append(float(np.min(d)))
    return min(results)

@st.cache_resource(show_spinner="تجهيز نموذج التعلّم الآلي على سيناريوهات افتراضية...")
def train_surrogate_model():
    """Train an ML surrogate on synthetic results from the simplified orbital simulator."""
    rng = np.random.default_rng(20261015)
    rows, targets = [], []
    # Synthetic data only; no real satellite ephemerides are used.
    for _ in range(1250):
        sat_alt = rng.uniform(350, 700)
        debris_alt = rng.uniform(300, 1500)
        phase = rng.uniform(0, 360)
        uncertainty = rng.uniform(0, 8)
        target = closest_approach(sat_alt, debris_alt, phase, uncertainty,
                                  horizon_min=HORIZON_MIN, n_time=61)
        rows.append([sat_alt, debris_alt, phase, uncertainty])
        targets.append(target)
    model = RandomForestRegressor(
        n_estimators=70, max_depth=14, min_samples_leaf=2,
        random_state=2026, n_jobs=-1
    )
    model.fit(np.asarray(rows), np.asarray(targets))
    return model

def predict_min_distance(model, sat_alt, debris_alt, phase, uncertainty):
    features = np.array([[sat_alt, debris_alt, phase % 360, uncertainty]], dtype=float)
    return float(max(0.0, model.predict(features)[0]))

def make_scene(debris_alt, debris_angle, maneuver, uncertainty):
    fig = go.Figure()
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 60)
    xx = EARTH_R*np.outer(np.cos(u), np.sin(v))
    yy = EARTH_R*np.outer(np.sin(u), np.sin(v))
    zz = EARTH_R*np.outer(np.ones_like(u), np.cos(v))
    surfacecolor = (zz / EARTH_R + 1) / 2
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz, surfacecolor=surfacecolor,
        colorscale=[[0,"#071d42"],[0.35,"#0c4c7c"],[0.65,"#1477a8"],[1,"#70d8f4"]],
        cmin=0, cmax=1, showscale=False, opacity=0.98,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.7, roughness=0.5),
        lightposition=dict(x=10000,y=5000,z=12000), name="Earth",
        hovertemplate="Earth<extra></extra>"
    ))
    for lat in [-60,-30,0,30,60]:
        t = np.linspace(0,2*np.pi,180)
        phi = np.radians(lat)
        rr = EARTH_R*np.cos(phi)
        fig.add_trace(go.Scatter3d(
            x=rr*np.cos(t), y=rr*np.sin(t), z=np.full_like(t,EARTH_R*np.sin(phi)),
            mode="lines", line=dict(color="rgba(115,194,232,.18)",width=1),
            showlegend=False, hoverinfo="skip"
        ))
    angles = np.linspace(0, 360, 361)
    def add_orbit(radius, color, name, width=3, dash="solid"):
        a = np.radians(angles)
        fig.add_trace(go.Scatter3d(
            x=radius*np.cos(a), y=radius*np.sin(a), z=np.zeros_like(a),
            mode="lines", name=name, line=dict(color=color,width=width,dash=dash),
            hovertemplate=name+"<extra></extra>"
        ))
    add_orbit(EARTH_R + SAT_ALT_DEFAULT, "#42e6a4", "المدار الأصلي")
    add_orbit(EARTH_R + SAT_ALT_DEFAULT + maneuver, "#58a6ff", "المدار الافتراضي بعد التغيير", 4)
    add_orbit(EARTH_R + debris_alt, "#ff667a", "مدار الحطام المفترض", 2, "dash")
    # The plotted points show a snapshot, not the predicted future position.
    a = np.radians(debris_angle)
    sat_r = EARTH_R + SAT_ALT_DEFAULT + maneuver
    debris_r = EARTH_R + debris_alt
    sat = (sat_r*np.cos(0), sat_r*np.sin(0), 0)
    debris = (debris_r*np.cos(a), debris_r*np.sin(a), 0)
    fig.add_trace(go.Scatter3d(
        x=[sat[0]],y=[sat[1]],z=[0],mode="markers",name="القمر الصناعي الافتراضي",
        marker=dict(size=8,color="#42e6a4",line=dict(color="white",width=1)),
        hovertemplate="القمر الصناعي الافتراضي<extra></extra>"
    ))
    fig.add_trace(go.Scatter3d(
        x=[debris[0]],y=[debris[1]],z=[0],mode="markers",name="المخلفات الفضائية",
        marker=dict(size=8,color="#ff667a",symbol="diamond",line=dict(color="white",width=1)),
        hovertemplate="المخلفات الفضائية<extra></extra>"
    ))
    # Snapshot line is intentionally labeled as illustrative, not future miss distance.
    fig.add_trace(go.Scatter3d(
        x=[sat[0],debris[0]],y=[sat[1],debris[1]],z=[0,0],
        mode="lines",name="مسافة هندسية في اللقطة",
        line=dict(color="#ffc66d",width=3,dash="dot")
    ))
    rng = np.random.default_rng(22)
    n = 260
    phi = rng.uniform(0,2*np.pi,n)
    costheta = rng.uniform(-1,1,n)
    rad = rng.uniform(EARTH_R*1.7, EARTH_R*2.3,n)
    stheta = np.arccos(costheta)
    fig.add_trace(go.Scatter3d(
        x=rad*np.sin(stheta)*np.cos(phi), y=rad*np.sin(stheta)*np.sin(phi),
        z=rad*np.cos(stheta), mode="markers", name="نجوم توضيحية",
        marker=dict(size=1.5,color="rgba(200,220,255,.65)"),
        showlegend=False, hoverinfo="skip"
    ))
    fig.update_layout(
        template="plotly_dark", height=610,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=45,b=0),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,
                    bgcolor="rgba(5,13,25,.5)",font=dict(size=10,color="#dceaff")),
        scene=dict(bgcolor="rgba(0,0,0,0)", aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.65,y=1.55,z=1.25))),
        title=dict(text="مشهد مداري توضيحي • ليس بيانات مدار حقيقية",x=0.02,
                   font=dict(size=13,color="#9dbbd8"))
    )
    return fig

# Sidebar
with st.sidebar:
    st.markdown("## 🛰️ ORBIT-SHIELD")
    st.caption("محاكاة ذكية • مختبر المخاطر المدارية")
    st.markdown("---")
    st.markdown("### مدخلات السيناريو")
    sat_alt = st.slider("ارتفاع القمر الصناعي الافتراضي (كم)", 350, 700, 500, 10)
    debris_alt = st.slider("ارتفاع الحطام (كم)", 300, 1500, 600, 10)
    debris_angle = st.slider("زاوية الحطام الابتدائية (°)", 0, 360, 20, 1)
    uncertainty = st.slider("عدم اليقين الزاوي (°)", 0.0, 8.0, 2.0, 0.5)
    horizon = st.select_slider("مدة التنبؤ الافتراضية", options=[30, 60, 90, 120], value=90,
                               format_func=lambda x: f"{x} دقيقة")
    maneuver = st.select_slider("التغيير الشعاعي الافتراضي (كم)", options=MANEUVERS, value=100)
    threshold = st.slider("عتبة تنبيه توضيحية (كم)", 10, 500, 100, 10)
    trials = st.slider("عدد تجارب اختبار الثبات", 20, 200, 100, 20)
    st.markdown("---")
    run_test = st.button("إعادة تنفيذ الاختبار", use_container_width=True)
    st.caption("بيانات مولّدة اصطناعيًا • نموذج بحثي مبسط")

if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if run_test:
    st.session_state.run_count += 1

st.markdown("""
<div class="hero">
  <span class="pill">AI-ASSISTED RESEARCH PROTOTYPE • SYNTHETIC DATA</span>
  <h1 style="margin:.65rem 0 .25rem 0;font-size:2.1rem;letter-spacing:.02em;">ORBIT-SHIELD</h1>
  <div style="font-size:1.05rem;color:#c5d8ec;">محاكاة ذكية لتجنب اصطدام الأقمار الصناعية الصغيرة بالمخلفات الفضائية باستخدام الذكاء الاصطناعي</div>
  <div class="small-muted" style="margin-top:.55rem;">نموذج أولي بحثي للتعلم والاختبار، وليس نظامًا تشغيليًا للأقمار الصناعية.</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("يجري تحليل السيناريو..."):
    model = train_surrogate_model()
    # Use the selected horizon for direct simulation; ML surrogate is trained at 90 minutes.
    base_exact = closest_approach(sat_alt, debris_alt, debris_angle, uncertainty, horizon, N_TIME)
    baseline_ml = predict_min_distance(model, sat_alt, debris_alt, debris_angle, uncertainty)
    candidate_options = []
    for m in MANEUVERS:
        candidate_alt = sat_alt + m
        if not 200 <= candidate_alt <= 2000:
            continue
        exact = closest_approach(candidate_alt, debris_alt, debris_angle, uncertainty, horizon, N_TIME)
        pred = predict_min_distance(model, candidate_alt, debris_alt, debris_angle, uncertainty)
        candidate_options.append({"التغيير الشعاعي الافتراضي (كم)": m,
                                  "المسافة الأقرب المحسوبة (كم)": exact,
                                  "تقدير نموذج التعلّم الآلي (كم)": pred})
    # Recommendation is only the best among the tested options for this simplified model.
    best = max(candidate_options, key=lambda row: row["المسافة الأقرب المحسوبة (كم)"])
    chosen_m = int(best["التغيير الشعاعي الافتراضي (كم)"])
    chosen_exact = float(best["المسافة الأقرب المحسوبة (كم)"])
    chosen_ml = float(best["تقدير نموذج التعلّم الآلي (كم)"])

    rng = np.random.default_rng(20261015 + st.session_state.run_count)
    selected_actions = []
    for _ in range(trials):
        d_alt = float(np.clip(debris_alt + rng.normal(0, min(30, max(1, debris_alt * .03))), 300, 1500))
        d_angle = float((debris_angle + rng.normal(0, uncertainty if uncertainty > 0 else .1)) % 360)
        # Re-select best tested action in each perturbed synthetic scenario.
        scores = []
        for m in MANEUVERS:
            c_alt = sat_alt + m
            if c_alt < 200:
                scores.append(-1.0)
            else:
                scores.append(closest_approach(c_alt, d_alt, d_angle, uncertainty, horizon, 61))
        selected_actions.append(MANEUVERS[int(np.argmax(scores))])
    counts = pd.Series(selected_actions).value_counts()
    stable_pct = 100 * counts.iloc[0] / len(selected_actions)
    distinct_actions = len(counts)
    model_label = "تنبيه توضيحي: المسافة أقل من العتبة" if chosen_exact < threshold else "المسافة أعلى من عتبة التنبيه التوضيحية"

# Main layout
left, right = st.columns([1.7, 1], gap="large")
with left:
    st.markdown("### المشهد المداري")
    st.plotly_chart(make_scene(debris_alt, debris_angle, chosen_m, uncertainty),
                    use_container_width=True, config={"displaylogo":False,"scrollZoom":True})
    st.caption("النقاط تمثل لقطة هندسية؛ المسافة الأقرب المتوقعة تُحسب منفصلًا ضمن نموذج مداري دائري مبسط.")
with right:
    st.markdown("### ملخص السيناريو")
    st.metric("أقرب مسافة محسوبة خلال المدة", f"{chosen_exact:,.1f} كم",
              delta=f"{chosen_exact-base_exact:+,.1f} كم مقارنة بالحالة دون تغيير")
    st.metric("الخيار الأفضل ضمن الخيارات المختبرة", f"{chosen_m:+} كم",
              help="أفضل خيار ضمن مجموعة تغييرات افتراضية محدودة في النموذج المبسط، وليس توصية مناورة حقيقية.")
    st.metric("ثبات الاختيار في التجارب", f"{stable_pct:.1f}%",
              help="نسبة التجارب التي اختارت الخيار الأكثر تكرارًا. لا تمثل موثوقية تشغيلية.")
    if chosen_exact < threshold:
        st.warning(f"{model_label}: {chosen_exact:,.1f} كم مقابل {threshold} كم.")
    else:
        st.success(f"{model_label}: {chosen_exact:,.1f} كم مقابل {threshold} كم.")
    st.markdown("<div class='note'>نموذج التعلّم الآلي يتعلم تقريب نتائج محاكاة اصطناعية مبسطة. لا يستخدم بيانات أقمار حقيقية ولا يثبت احتمال اصطدام فعلي.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## اختبار ثبات القرار")
st.write("يغيّر الاختبار زاوية الحطام وارتفاعه ضمن نطاقات افتراضية، ثم يعيد مقارنة الخيارات نفسها. الهدف هو معرفة مدى تغيّر اختيار النموذج عندما تتغير المدخلات.")
c1, c2, c3 = st.columns(3)
c1.metric("عدد السيناريوهات المختبرة", f"{trials}")
c2.metric("عدد الخيارات المختلفة التي ظهرت", f"{distinct_actions}")
c3.metric("نسبة الخيار الأكثر تكرارًا", f"{stable_pct:.1f}%")
st.caption("هذه النسبة مقياس ثبات داخل التجربة المحددة فقط، وليست نسبة نجاح أو ضمانًا لتجنب الاصطدام.")

st.markdown("## مقارنة الخيارات الافتراضية")
comparison = pd.DataFrame(candidate_options)
comparison["المسافة الأقرب المحسوبة (كم)"] = comparison["المسافة الأقرب المحسوبة (كم)"].round(2)
comparison["تقدير نموذج التعلّم الآلي (كم)"] = comparison["تقدير نموذج التعلّم الآلي (كم)"].round(2)
st.dataframe(comparison, use_container_width=True, hide_index=True)

st.markdown("## كيف نقرأ النتيجة؟")
st.markdown("""
- **المسافة الأقرب المحسوبة:** أقل مسافة ينتجها النموذج الهندسي خلال مدة التنبؤ المحددة.
- **تقدير نموذج التعلّم الآلي:** تقدير يتعلمه نموذج Random Forest من سيناريوهات اصطناعية أنشأها المحاكي نفسه.
- **الخيار الأفضل ضمن الخيارات المختبرة:** الخيار الذي حقق أكبر مسافة أقرب محسوبة ضمن مجموعة محدودة من التغييرات الافتراضية؛ لا يعني أنه آمن عمليًا.
- **ثبات الاختيار:** يوضح مدى تكرار اختيار الإجراء نفسه عند تغيير المدخلات، وليس دقة التنبؤ أو موثوقية التشغيل الحقيقي.
""")

with st.expander("افتراضات النموذج وحدوده"):
    st.markdown("""
    1. يستخدم النموذج مدارين دائريين وفي مستوى واحد، مع تقريب ثنائي الجسم لحساب السرعة الزاوية.
    2. مدة التنبؤ قصيرة ومحددة، والحساب يعتمد على نقاط زمنية متقطعة؛ لذلك قد لا يلتقط كل تفاصيل الاقتراب الحقيقي.
    3. نموذج التعلّم الآلي مدرّب على بيانات اصطناعية مولّدة من المحاكي، وليس على سجلات أقمار صناعية حقيقية.
    4. التغيير الشعاعي الافتراضي ليس مناورة دفع واقعية مثبتة، ولا يتضمن الوقود أو قيود المركبة أو تخطيط مناورة فعليًا.
    5. عتبة التنبيه قابلة للتغيير لأغراض تعليمية وليست معيار أمان رسميًا.
    6. يلزم التحقق من النموذج باستخدام بيانات مدارية موثوقة وطرق مستقلة قبل أي استنتاج عن الأداء الواقعي.
    """)

st.markdown("---")
st.markdown("<div class='small-muted'>ORBIT-SHIELD • نموذج بحثي أولي • محاكاة تعليمية واختبار منهجي</div>", unsafe_allow_html=True)
