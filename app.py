import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ORBIT-SHIELD | Orbital Risk Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter','IBM Plex Sans Arabic',sans-serif; }
.stApp { background:
  radial-gradient(ellipse at 75% 0%, rgba(17,75,120,.22), transparent 38%),
  linear-gradient(180deg,#07101d 0%,#030711 100%); color:#eaf2ff; }
.block-container { padding-top: 1.5rem; max-width: 1500px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1728,#07101c); border-right:1px solid #1c3048; }
div[data-testid="stMetric"] { background:rgba(13,29,49,.75); border:1px solid #203c58; padding:15px 18px; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:#8faac7; }
div[data-testid="stMetricValue"] { color:#f4f8ff; }
div.stButton > button { border-radius:10px; border:1px solid #2a83b8; background:#0b6e9f; color:white; font-weight:700; }
.small-muted { color:#91a9c4; font-size:.9rem; }
.hero { padding: 22px 24px; border:1px solid #203c58; border-radius:18px;
 background:linear-gradient(110deg,rgba(11,36,62,.96),rgba(6,15,29,.88)); margin-bottom:16px; }
.pill { display:inline-block; border:1px solid #27617d; border-radius:99px; padding:4px 10px; color:#76d8ff; font-size:12px; letter-spacing:.08em; }
</style>
""", unsafe_allow_html=True)

EARTH_R = 6371.0
SAT_ALT = 500.0
MANEUVERS = [-200, -100, 0, 100, 200]

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("## 🛰️ ORBIT-SHIELD")
    st.caption("ORBITAL RISK LAB  /  SIMULATION")
    st.markdown("---")
    st.markdown("### Scenario controls")
    debris_alt = st.slider("Debris altitude (km)", 300, 1500, 600, 10)
    debris_angle = st.slider("Debris angle (°)", 0, 360, 20, 1)
    uncertainty = st.slider("Angular uncertainty (°)", 0.0, 8.0, 2.0, 0.5)
    maneuver = st.select_slider("Hypothetical radial change (km)", options=MANEUVERS, value=100)
    threshold = st.slider("Illustrative alert threshold (km)", 500, 5000, 2000, 100)
    trials = st.slider("Robustness trials", 20, 300, 100, 20)
    st.markdown("---")
    run_test = st.button("Run risk & robustness check", use_container_width=True)
    st.caption("Prototype model • simplified 2D geometry")

if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if run_test:
    st.session_state.run_count += 1

# ---------- Geometry helpers ----------
def xyz_circle(radius, angles_deg, z=0.0):
    a = np.radians(np.asarray(angles_deg))
    return radius*np.cos(a), radius*np.sin(a), np.zeros_like(a) + z

def separation(r1, r2, angle_deg):
    a = np.radians(angle_deg)
    return float(np.sqrt((r1-r2*np.cos(a))**2 + (r2*np.sin(a))**2))

def make_globe_and_orbits(altitude, angle, radial_change, uncertainty_deg):
    fig = go.Figure()

    # Sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 60)
    xx = EARTH_R*np.outer(np.cos(u), np.sin(v))
    yy = EARTH_R*np.outer(np.sin(u), np.sin(v))
    zz = EARTH_R*np.outer(np.ones_like(u), np.cos(v))
    # Blue ocean gradient gives a clean globe; labels explicitly describe the prototype.
    surfacecolor = (zz / EARTH_R + 1) / 2
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz, surfacecolor=surfacecolor,
        colorscale=[[0,"#071d42"],[0.35,"#0c4c7c"],[0.65,"#1477a8"],[1,"#70d8f4"]],
        cmin=0, cmax=1, showscale=False, opacity=0.98,
        lighting=dict(ambient=0.45, diffuse=0.85, specular=0.7, roughness=0.5),
        lightposition=dict(x=10000,y=5000,z=12000), name="Earth"
    ))

    # Subtle latitude rings
    for lat in [-60,-30,0,30,60]:
        t = np.linspace(0,2*np.pi,180)
        phi = np.radians(lat)
        rr = EARTH_R* np.cos(phi)
        fig.add_trace(go.Scatter3d(
            x=rr*np.cos(t), y=rr*np.sin(t), z=np.full_like(t,EARTH_R*np.sin(phi)),
            mode="lines", line=dict(color="rgba(115,194,232,.18)",width=1),
            showlegend=False, hoverinfo="skip"
        ))

    orbit_angles = np.linspace(0,360,361)
    def add_orbit(radius, color, name, width=3, dash="solid"):
        x,y,z = xyz_circle(radius, orbit_angles)
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="lines",name=name,
            line=dict(color=color,width=width,dash=dash), hoverinfo="name"))

    add_orbit(EARTH_R+SAT_ALT, "#42e6a4", "Original satellite orbit", 3)
    add_orbit(EARTH_R+SAT_ALT+radial_change, "#58a6ff", "Candidate orbit", 4)
    add_orbit(EARTH_R+altitude, "#ff667a", "Debris reference orbit", 2, "dash")

    # Satellite point and debris point in 3D equatorial plane
    sat_r = EARTH_R + SAT_ALT + radial_change
    a = np.radians(angle)
    sat = (sat_r*np.cos(a), sat_r*np.sin(a), 0)
    debris = ((EARTH_R+altitude)*np.cos(a+np.radians(uncertainty_deg)),
              (EARTH_R+altitude)*np.sin(a+np.radians(uncertainty_deg)), 0)
    fig.add_trace(go.Scatter3d(x=[sat[0]],y=[sat[1]],z=[sat[2]],mode="markers",
        name="Satellite", marker=dict(size=7,color="#42e6a4",line=dict(color="white",width=1))))
    fig.add_trace(go.Scatter3d(x=[debris[0]],y=[debris[1]],z=[debris[2]],mode="markers",
        name="Debris", marker=dict(size=7,color="#ff667a",symbol="diamond",line=dict(color="white",width=1))))
    fig.add_trace(go.Scatter3d(x=[sat[0],debris[0]],y=[sat[1],debris[1]],z=[0,0],
        mode="lines",name="Illustrative separation",line=dict(color="#ffc66d",width=3,dash="dot")))

    # Star field
    rng = np.random.default_rng(22)
    n = 260
    phi = rng.uniform(0,2*np.pi,n)
    costheta = rng.uniform(-1,1,n)
    rad = rng.uniform(EARTH_R*1.7, EARTH_R*2.3,n)
    stheta = np.arccos(costheta)
    fig.add_trace(go.Scatter3d(
        x=rad*np.sin(stheta)*np.cos(phi),
        y=rad*np.sin(stheta)*np.sin(phi),
        z=rad*np.cos(stheta), mode="markers", name="Stars",
        marker=dict(size=1.5,color="rgba(200,220,255,.65)"),
        showlegend=False, hoverinfo="skip"
    ))

    fig.update_layout(
        template="plotly_dark", height=650,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=35,b=0),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,
                    bgcolor="rgba(5,13,25,.5)",font=dict(size=10,color="#dceaff")),
        scene=dict(
            bgcolor="rgba(0,0,0,0)", aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.65,y=1.55,z=1.25))
        ),
        title=dict(text="ORBITAL SCENE  /  SIMULATED GEOMETRY",x=0.02,font=dict(size=13,color="#9dbbd8"))
    )
    return fig

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <span class="pill">SPACE SITUATIONAL AWARENESS • PROTOTYPE</span>
  <h1 style="margin:.65rem 0 .25rem 0;font-size:2.1rem;letter-spacing:.02em;">ORBIT-SHIELD</h1>
  <div style="font-size:1.05rem;color:#c5d8ec;">نظام محاكاة لتقييم الاقتراب من المخلفات الفضائية واختبار ثبات القرار</div>
  <div class="small-muted" style="margin-top:.55rem;">Interactive orbital-risk research demonstrator · not an operational flight system</div>
</div>
""", unsafe_allow_html=True)

r0 = EARTH_R + SAT_ALT
rd = EARTH_R + debris_alt
base_dist = separation(r0, rd, debris_angle)
candidate_dist = separation(r0 + maneuver, rd, debris_angle + uncertainty)
worst_case = min(separation(r0 + maneuver, rd, debris_angle + p)
                 for p in [-uncertainty, 0, uncertainty])
baseline_worst = min(separation(r0, rd, debris_angle + p)
                     for p in [-uncertainty, 0, uncertainty])

# Deterministic robustness sample: perturb angle and debris altitude, compare selected action
rng = np.random.default_rng(20261014)
perturb_angles = rng.uniform(-uncertainty, uncertainty, trials)
perturb_alts = rng.uniform(-min(30, max(1, debris_alt*0.03)),
                           min(30, max(1, debris_alt*0.03)), trials)
actions = []
for da, dh in zip(perturb_angles, perturb_alts):
    drr = EARTH_R + debris_alt + dh
    # Evaluate candidate maneuvers under the same perturbed input; choose maximum separation.
    scores = [separation(r0+m, drr, debris_angle+da) for m in MANEUVERS]
    actions.append(MANEUVERS[int(np.argmax(scores))])
stable_pct = 100 * (sum(a == actions[0] for a in actions) / len(actions)) if actions else 0
unique_actions = len(set(actions))
alert = "Elevated in this model" if candidate_dist < threshold else "Below illustrative threshold"

# Main dashboard
left, right = st.columns([1.8,1], gap="large")
with left:
    st.markdown("### Orbital scene")
    st.plotly_chart(make_globe_and_orbits(debris_alt,debris_angle,maneuver,uncertainty),
                    use_container_width=True, config={"displaylogo":False,"scrollZoom":True})
    st.caption("Visual scale is illustrative. The globe and circular tracks are a conceptual 3D scene, not real ephemerides.")
with right:
    st.markdown("### Scenario summary")
    st.metric("Geometric separation", f"{candidate_dist:,.1f} km",
              delta=f"{candidate_dist-base_dist:+,.1f} km vs. unmodified geometry")
    st.metric("Chosen radial change", f"{maneuver:+} km")
    st.metric("Robustness sample", f"{stable_pct:.0f}%", help="Fraction of perturbed samples choosing the same action as the first sample; not real-world reliability.")
    st.markdown("#### Model status")
    if candidate_dist < threshold:
        st.warning(f"Alert threshold crossed: {candidate_dist:,.1f} km is below the illustrative {threshold:,} km setting.")
    else:
        st.success(f"Below the illustrative alert threshold of {threshold:,} km.")
    st.info("This alert is based on simplified geometry only. It does not establish a real collision probability or maneuver safety.")

st.markdown("---")
st.markdown("## Decision reliability test")
st.write("This section perturbs the input angle and debris altitude, then checks how consistently the simplified rule chooses the same radial-change option.")
c1,c2,c3 = st.columns(3)
c1.metric("Perturbed scenarios", f"{trials}")
c2.metric("Different actions selected", f"{unique_actions}")
c3.metric("Same-action fraction", f"{stable_pct:.1f}%")

dist_table = pd.DataFrame([
    {"Measure":"Unmodified geometry separation","Distance (km)":round(base_dist,2)},
    {"Measure":"After selected hypothetical change","Distance (km)":round(candidate_dist,2)},
    {"Measure":"Worst tested separation for selected change","Distance (km)":round(worst_case,2)},
    {"Measure":"Unmodified worst tested separation","Distance (km)":round(baseline_worst,2)},
])
st.dataframe(dist_table, use_container_width=True, hide_index=True)

st.markdown("### How to interpret the result")
st.markdown("""
- **Geometric separation** is the distance between two points in this simplified snapshot.
- **Robustness sample** describes how often this toy decision rule repeats the same choice after small input changes.
- **It is not an operational collision probability.** Real assessment requires time-dependent orbital propagation, uncertainty/covariance, relative velocity, and validated maneuver constraints.
""")

with st.expander("Model assumptions and limitations"):
    st.markdown("""
    1. The Earth and orbits are visualized as a conceptual 3D scene with circular paths.
    2. The calculation is a simplified geometric snapshot, not a time-propagated closest-approach calculation.
    3. A radial change is a hypothetical comparison option, not a physically validated thruster maneuver.
    4. The alert threshold is a user-controlled illustrative setting, not a real safety standard.
    5. The robustness percentage measures consistency of this simple rule under sampled perturbations; it does not validate real-world reliability.
    """)

st.markdown("---")
st.markdown("<div class='small-muted'>ORBIT-SHIELD • Research prototype • For educational simulation and method evaluation</div>", unsafe_allow_html=True)
