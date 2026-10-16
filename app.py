import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ORBIT-SHIELD", page_icon="🛰️", layout="wide")

EARTH_R = 6371.0
MU = 398600.4418
OPTIONS_KM = [-200, -100, 0, 100, 200]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family:'IBM Plex Sans Arabic',sans-serif;}
.stApp {background:linear-gradient(180deg,#07101e,#030711);color:#eef5ff;}
.block-container {max-width:1450px;padding-top:1.2rem;}
div[data-testid="stMetric"] {background:#0c1a2b;border:1px solid #213c58;padding:14px;border-radius:13px;}
div.stButton>button, div[data-testid="stFormSubmitButton]>button {background:#5145e5;color:white;border:0;border-radius:10px;font-weight:700;width:100%;}
section[data-testid="stSidebar"] {background:#0a1424;border-right:1px solid #243c57;}
.hero {padding:20px 24px;border:1px solid #24415e;border-radius:17px;background:linear-gradient(110deg,#102b49,#07101d);margin-bottom:18px;}
.muted {color:#9bb1ca;font-size:.92rem;}
.monitor {padding:18px;border:1px solid #1d4b80;border-radius:15px;background:#061321;text-align:center;}
</style>
""", unsafe_allow_html=True)

def mean_motion(alt_km):
    r = EARTH_R + float(alt_km)
    return np.sqrt(MU / r**3)

def closest_approach(sat_alt, debris_alt, phase_deg, angle_uncertainty_deg, duration_min, samples=181):
    """Simplified coplanar circular-orbit model; synthetic educational approximation."""
    t = np.linspace(0, duration_min * 60, samples)
    rs = EARTH_R + sat_alt
    rd = EARTH_R + debris_alt
    relative_rate = mean_motion(debris_alt) - mean_motion(sat_alt)
    tested = []
    for offset in (-angle_uncertainty_deg, 0.0, angle_uncertainty_deg):
        phase = np.radians(phase_deg + offset) + relative_rate * t
        d = np.sqrt(np.maximum(0, rs**2 + rd**2 - 2*rs*rd*np.cos(phase)))
        tested.append(float(np.min(d)))
    return min(tested)

def build_scene(sat_alt, debris_alt, angle_deg, maneuver_km):
    fig = go.Figure()
    # Stylized Earth sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 60)
    x = EARTH_R*np.outer(np.cos(u), np.sin(v))
    y = EARTH_R*np.outer(np.sin(u), np.sin(v))
    z = EARTH_R*np.outer(np.ones_like(u), np.cos(v))
    col = (z/EARTH_R + 1)/2
    fig.add_trace(go.Surface(x=x,y=y,z=z,surfacecolor=col,
        colorscale=[[0,"#061c40"],[.35,"#0b4e82"],[.7,"#1379b1"],[1,"#70dff5"]],
        cmin=0,cmax=1,showscale=False,opacity=.98,
        lighting=dict(ambient=.5,diffuse=.85,specular=.65,roughness=.5),
        name="الأرض",hovertemplate="الأرض (رسم توضيحي)<extra></extra>"))
    # Latitude lines
    for lat in [-60,-30,0,30,60]:
        th=np.linspace(0,2*np.pi,180); p=np.radians(lat); rr=EARTH_R*np.cos(p)
        fig.add_trace(go.Scatter3d(x=rr*np.cos(th),y=rr*np.sin(th),
            z=np.full_like(th,EARTH_R*np.sin(p)),mode="lines",
            line=dict(color="rgba(120,190,230,.18)",width=1),showlegend=False,hoverinfo="skip"))
    ang=np.linspace(0,2*np.pi,361)
    def orbit(alt,color,name,width=3,dash="solid"):
        rr=EARTH_R+alt
        fig.add_trace(go.Scatter3d(x=rr*np.cos(ang),y=rr*np.sin(ang),z=np.zeros_like(ang),
            mode="lines",name=name,line=dict(color=color,width=width,dash=dash)))
    orbit(sat_alt,"#36d99a","المسار الأصلي")
    orbit(sat_alt+maneuver_km,"#5aa8ff","المسار بعد التغيير الافتراضي",4)
    orbit(debris_alt,"#ff6477","مسار المخلفات",2,"dash")
    sr=EARTH_R+sat_alt+maneuver_km; dr=EARTH_R+debris_alt; a=np.radians(angle_deg)
    fig.add_trace(go.Scatter3d(x=[sr],y=[0],z=[0],mode="markers",name="القمر الصناعي",
        marker=dict(size=7,color="#36d99a",line=dict(color="white",width=1))))
    fig.add_trace(go.Scatter3d(x=[dr*np.cos(a)],y=[dr*np.sin(a)],z=[0],mode="markers",name="المخلفات الفضائية",
        marker=dict(size=8,color="#ff6477",symbol="diamond",line=dict(color="white",width=1))))
    # Illustrative stars
    rng=np.random.default_rng(15); n=240
    phi=rng.uniform(0,2*np.pi,n); ct=rng.uniform(-1,1,n); rad=rng.uniform(EARTH_R*1.65,EARTH_R*2.1,n)
    stheta=np.arccos(ct)
    fig.add_trace(go.Scatter3d(x=rad*np.sin(stheta)*np.cos(phi),y=rad*np.sin(stheta)*np.sin(phi),
        z=rad*np.cos(stheta),mode="markers",name="نجوم توضيحية",
        marker=dict(size=1.5,color="rgba(205,225,255,.7)"),showlegend=False,hoverinfo="skip"))
    fig.update_layout(template="plotly_dark",height=540,margin=dict(l=0,r=0,t=45,b=0),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=1.03,x=0,font=dict(size=10)),
        scene=dict(bgcolor="rgba(0,0,0,0)",aspectmode="data",
            xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.55,y=1.55,z=1.15))),
        title=dict(text="المشهد المداري ثلاثي الأبعاد • تمثيل توضيحي",x=.02,font=dict(size=13)))
    return fig

if "scenario" not in st.session_state:
    st.session_state.scenario = {
        "sat_alt":500, "debris_alt":600, "angle":20, "maneuver":100,
        "uncertainty":2, "threshold":100, "duration":90, "trials":100
    }

st.markdown("""
<div class="hero">
 <div style="color:#79d9ff;font-size:.85rem;">SIMULATION • SYNTHETIC SCENARIO</div>
 <h1 style="margin:.35rem 0;">ORBIT-SHIELD 🛰️</h1>
 <div style="font-size:1.08rem;color:#d1e0ef;">محاكاة ذكية لتجنب اصطدام الأقمار الصناعية الصغيرة بالمخلفات الفضائية باستخدام الذكاء الاصطناعي</div>
 <div class="muted" style="margin-top:.6rem;">نموذج بحثي مبسّط للاختبار والتعلّم، وليس نظامًا تشغيليًا للملاحة الفضائية.</div>
</div>
""", unsafe_allow_html=True)

# Controls matching the requested reference design
with st.sidebar:
    st.markdown("## 🛰️ لوحة التحكم")
    st.caption("غيّري القيم ثم اضغطي تحديث المحاكاة")
    with st.form("orbit_controls"):
        sat_alt = st.slider("ارتفاع القمر الصناعي الافتراضي (كم)",350,700,
                            int(st.session_state.scenario["sat_alt"]),10)
        debris_alt = st.slider("ارتفاع المخلفات (كم)",300,1500,
                               int(st.session_state.scenario["debris_alt"]),10)
        angle = st.slider("زاوية المخلفات (درجة)",0,360,
                          int(st.session_state.scenario["angle"]),1)
        maneuver = st.slider("تغيير نصف القطر الافتراضي (كم)",-200,200,
                             int(st.session_state.scenario["maneuver"]),10)
        uncertainty = st.slider("عدم اليقين الزاوي (درجة)",0.0,8.0,
                                float(st.session_state.scenario["uncertainty"]),0.5)
        duration = st.select_slider("مدة التنبؤ",options=[30,60,90,120],
                                    value=int(st.session_state.scenario["duration"]),
                                    format_func=lambda x:f"{x} دقيقة")
        threshold = st.slider("عتبة التنبيه التوضيحية (كم)",10,500,
                              int(st.session_state.scenario["threshold"]),10)
        trials = st.select_slider("عدد تجارب اختبار الثبات",options=[20,40,60,80,100,150,200],
                                  value=int(st.session_state.scenario["trials"]))
        submitted = st.form_submit_button("تحديث المحاكاة")
    if submitted:
        st.session_state.scenario = {"sat_alt":sat_alt,"debris_alt":debris_alt,
            "angle":angle,"maneuver":maneuver,"uncertainty":uncertainty,
            "duration":duration,"threshold":threshold,"trials":trials}
        st.rerun()
    st.markdown("---")
    st.caption("كل القيم افتراضية وقابلة للتعديل لأغراض البحث.")

s=st.session_state.scenario
sat_alt=s["sat_alt"]; debris_alt=s["debris_alt"]; angle=s["angle"]
maneuver=s["maneuver"]; uncertainty=s["uncertainty"]; duration=s["duration"]
threshold=s["threshold"]; trials=s["trials"]

# Compare available hypothetical maneuvers and separately evaluate the selected control.
rows=[]
for m in OPTIONS_KM:
    if sat_alt+m < 200: 
        continue
    d=closest_approach(sat_alt+m,debris_alt,angle,uncertainty,duration)
    rows.append({"التغيير الافتراضي (كم)":m,"أقرب مسافة محسوبة (كم)":d})
comparison=pd.DataFrame(rows)
selected_distance=closest_approach(sat_alt+maneuver,debris_alt,angle,uncertainty,duration)
baseline_distance=closest_approach(sat_alt,debris_alt,angle,uncertainty,duration)
best_row=comparison.loc[comparison["أقرب مسافة محسوبة (كم)"].idxmax()]
best_m=int(best_row["التغيير الافتراضي (كم)"])
best_distance=float(best_row["أقرب مسافة محسوبة (كم)"])

# Stability: choose the best of the same finite options in perturbed input scenarios.
rng=np.random.default_rng(20261015)
actions=[]
for _ in range(trials):
    alt2=float(np.clip(debris_alt+rng.normal(0,15),300,1500))
    angle2=float((angle+rng.normal(0,max(uncertainty,.25)))%360)
    scores=[]
    for m in OPTIONS_KM:
        if sat_alt+m < 200:
            scores.append(-1)
        else:
            scores.append(closest_approach(sat_alt+m,alt2,angle2,uncertainty,duration,samples=91))
    actions.append(OPTIONS_KM[int(np.argmax(scores))])
freq=pd.Series(actions).value_counts()
stable_pct=100*freq.iloc[0]/trials
unique_actions=len(freq)
most_common=int(freq.index[0])
risk_label="تنبيه توضيحي: أقل من العتبة" if selected_distance < threshold else "أعلى من عتبة التنبيه التوضيحية"

left,right=st.columns([1.55,1],gap="large")
with left:
    st.markdown("### المشهد المداري")
    st.plotly_chart(build_scene(sat_alt,debris_alt,angle,maneuver),use_container_width=True,
                    config={"displaylogo":False,"scrollZoom":True})
    st.caption("الرسم يوضح المدارات والمواضع بشكل تخطيطي؛ المسافة المحسوبة تُستخرج من نموذج منفصل.")
with right:
    st.markdown("### ملخص السيناريو")
    st.metric("المسافة الأقرب المحسوبة مع التغيير المحدد",f"{selected_distance:,.2f} كم",
              delta=f"{selected_distance-baseline_distance:+,.2f} كم مقارنة دون تغيير")
    st.metric("الخيار الأفضل بين الخيارات المختبرة",f"{best_m:+} كم",
              help="أفضل خيار في النموذج المبسّط فقط، وليس توصية مناورة فعلية.")
    st.metric("ثبات الاختيار عبر السيناريوهات",f"{stable_pct:.1f}%",
              help="نسبة تكرار الخيار الأكثر شيوعًا في الاختبار المحدد، وليست احتمال تجنب الاصطدام.")
    if selected_distance < threshold:
        st.warning(risk_label)
    else:
        st.success(risk_label)
    st.markdown(f"""
    <div class="monitor">
      <div style="color:#39d9ef;font-weight:700;font-size:1.05rem;">ORBIT-SHIELD • STATUS MONITOR</div>
      <hr style="border-color:#234363">
      <div>المسافة الحالية المحسوبة: <b>{selected_distance:,.2f} كم</b></div>
      <div>عتبة التنبيه التوضيحية: <b>{threshold} كم</b></div>
      <div style="margin-top:8px;color:{'#ff687a' if selected_distance < threshold else '#36d99a'};font-weight:700;">
        {'⚠ يحتاج السيناريو إلى مراجعة ضمن النموذج' if selected_distance < threshold else '✓ أعلى من عتبة التنبيه المحددة'}
      </div>
      <div style="color:#91a9c4;font-size:.8rem;margin-top:8px;">لا تمثل الحالة حكم أمان واقعيًا</div>
    </div>
    """,unsafe_allow_html=True)

st.markdown("---")
st.markdown("## اختبار ثبات القرار")
a,b,c=st.columns(3)
a.metric("عدد السيناريوهات",trials)
b.metric("عدد الخيارات التي اختيرت",unique_actions)
c.metric("الخيار الأكثر تكرارًا",f"{most_common:+} كم")
if stable_pct >= 80:
    st.info(f"في هذه التجربة المحددة، كان الخيار الأكثر تكرارًا هو {most_common:+} كم بنسبة {stable_pct:.1f}%. هذا يصف ثبات الاختيار في السيناريوهات المختبرة فقط.")
else:
    st.warning(f"تغيّر اختيار النظام بين عدة خيارات؛ الخيار الأكثر تكرارًا كان {most_common:+} كم بنسبة {stable_pct:.1f}%. قد يشير ذلك إلى حساسية للمدخلات ضمن هذا النموذج.")
st.caption("اختبار الثبات يقيس تكرار القرار بعد تغيير افتراضي للمدخلات؛ ولا يقيس دقة التنبؤ أو احتمال الاصطدام الحقيقي.")

st.markdown("## مقارنة الخيارات الافتراضية")
comparison["أقرب مسافة محسوبة (كم)"]=comparison["أقرب مسافة محسوبة (كم)"].round(2)
st.dataframe(comparison,use_container_width=True,hide_index=True)

with st.expander("كيف تُحسب النتائج؟ وما حدود النموذج؟"):
    st.markdown("""
    - يستخدم الحساب مدارين دائريين في مستوى واحد، مع تقريب ثنائي الجسم لتقدير السرعة الزاوية.
    - تُحسب أقل مسافة على نقاط زمنية متفرقة خلال مدة التنبؤ المختارة.
    - تغيير نصف القطر قيمة افتراضية في النموذج، وليس مناورة دفع عملية مؤكدة.
    - عتبة التنبيه يحددها المستخدم لأغراض التجربة، وليست معيارًا رسميًا لسلامة الأقمار.
    - اختبار الثبات يغيّر ارتفاع الحطام وزاويته افتراضيًا ويعيد اختيار أفضل قيمة من مجموعة محدودة.
    - هذا الإصدار لا يستخدم بيانات مدارية حقيقية، ولا يثبت أن قرارًا ما سيمنع اصطدامًا في الواقع.
    """)

st.markdown("<div class='muted'>ORBIT-SHIELD • نموذج بحثي أولي قابل للتطوير والتحقق</div>",unsafe_allow_html=True)
