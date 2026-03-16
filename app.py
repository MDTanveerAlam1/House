import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─── Page Config (MUST be first Streamlit call) ──────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Outfit', sans-serif; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #f0f0f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(168,85,247,0.3));
    border: 1px solid rgba(99,102,241,0.5);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero-banner h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-banner p {
    color: #c4b5fd;
    font-size: 1.1rem;
    margin-top: 10px;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,102,241,0.3);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.8rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(52,211,153,0.25));
    border: 2px solid rgba(99,102,241,0.6);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    margin: 20px 0;
    animation: pulse-border 2s infinite;
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(99,102,241,0.6); }
    50%       { border-color: rgba(52,211,153,0.8); }
}
.result-price {
    font-size: 3rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-sub {
    color: #9ca3af;
    font-size: 1rem;
    margin-top: 8px;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #a78bfa;
    border-left: 4px solid #6366f1;
    padding-left: 14px;
    margin: 28px 0 16px 0;
}

/* Info tag */
.info-tag {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid rgba(99,102,241,0.5);
    color: #a78bfa;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

/* Feature badge */
.feature-badge {
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 8px;
    padding: 8px 14px;
    margin: 4px 0;
    font-size: 0.85rem;
    color: #6ee7b7;
}

/* Streamlit overrides */
.stSlider > div > div > div { background: #6366f1 !important; }
div[data-testid="stSelectbox"] > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.15) !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.6rem 2.5rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
}
.stTabs [data-baseweb="tab"] {
    color: #9ca3af !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Data ────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("house_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, scaler, info

@st.cache_data
def load_data():
    return pd.read_csv("Housing.csv")

model, scaler, model_info = load_model()
df_raw = load_data()
FEATURES = model_info["features"]


# ─── Prediction Helper ────────────────────────────────────────
def predict_price(inputs: dict) -> float:
    """Prepare features and return predicted price."""
    d = dict(inputs)
    area      = float(d['area'])
    bedrooms  = float(d['bedrooms'])
    bathrooms = float(d['bathrooms'])

    d['total_rooms']   = bedrooms + bathrooms
    d['area_per_room'] = area / (d['total_rooms'] + 1)
    d['luxury_score']  = d['airconditioning'] + d['hotwaterheating'] + d['prefarea'] + d['guestroom']
    d['log_area']      = np.log1p(area)
    d['has_parking']   = 1 if int(d['parking']) > 0 else 0

    X = pd.DataFrame([d])[FEATURES]
    return float(model.predict(X)[0])


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 House Predictor")
    st.markdown("---")

    st.markdown("### 📊 Model Performance")
    perf = model_info["performance"]
    st.markdown(f"""
    <div class="feature-badge">🎯 R² Score &nbsp;&nbsp; <b>{perf['R2']:.4f}</b></div>
    <div class="feature-badge">📉 MAE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b>₹{perf['MAE']:,.0f}</b></div>
    <div class="feature-badge">📐 RMSE &nbsp;&nbsp;&nbsp;&nbsp; <b>₹{perf['RMSE']:,.0f}</b></div>
    <div class="feature-badge">📏 MAPE &nbsp;&nbsp;&nbsp;&nbsp; <b>{perf['MAPE']:.2f}%</b></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Model Details")
    params = model_info["best_params"]
    st.markdown(f"""
    <div style='font-size:0.82rem; color:#c4b5fd; line-height:1.8'>
    <b>Model:</b> Random Forest<br>
    <b>Trees:</b> {params['n_estimators']}<br>
    <b>Max Depth:</b> {params['max_depth']}<br>
    <b>Min Split:</b> {params['min_samples_split']}<br>
    <b>Min Leaf:</b> {params['min_samples_leaf']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Dataset Info")
    st.markdown(f"""
    <div style='font-size:0.82rem; color:#c4b5fd; line-height:1.8'>
    <b>Total Houses:</b> {len(df_raw)}<br>
    <b>Features:</b> {df_raw.shape[1] - 1}<br>
    <b>Price Range:</b><br>
    ₹{df_raw['price'].min()/1e6:.2f}M – ₹{df_raw['price'].max()/1e6:.2f}M<br>
    <b>Avg Price:</b> ₹{df_raw['price'].mean()/1e6:.2f}M
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + Scikit-learn")


# ═══════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🏠 House Price Predictor</h1>
    <p>Enter your house details below and get an instant AI-powered price estimate</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Explorer", "📈 Model Insights"])


# ───────────────────────────────────────────────────────────────
# TAB 1 — PREDICT
# ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🏡 Enter House Details</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📐 Size & Structure**")
        area      = st.slider("Area (sq ft)", 1000, 16200, 5000, step=100)
        bedrooms  = st.selectbox("Bedrooms",  [1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4],        index=1)
        stories   = st.selectbox("Stories",   [1, 2, 3, 4],        index=1)
        parking   = st.selectbox("Parking Spots", [0, 1, 2, 3],    index=1)

    with col2:
        st.markdown("**🏘️ Location & Access**")
        mainroad = st.radio("Main Road Access?",
                            ["Yes", "No"], horizontal=True)
        prefarea = st.radio("Preferred Area?",
                            ["Yes", "No"], horizontal=True)

        st.markdown("**🛋️ Rooms**")
        guestroom = st.radio("Guest Room?",
                             ["Yes", "No"], horizontal=True)
        basement  = st.radio("Basement?",
                             ["Yes", "No"], horizontal=True)

    with col3:
        st.markdown("**⚡ Amenities**")
        airconditioning = st.radio("Air Conditioning?",
                                   ["Yes", "No"], horizontal=True)
        hotwaterheating = st.radio("Hot Water Heating?",
                                   ["Yes", "No"], horizontal=True)
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ["Furnished", "Semi-Furnished", "Unfurnished"]
        )

        # Live summary
        st.markdown("**📋 Summary**")
        furnish_short = {"Furnished": "✅ Furnished",
                         "Semi-Furnished": "🟡 Semi",
                         "Unfurnished": "❌ Unfurnished"}[furnishingstatus]
        ac_icon = "❄️" if airconditioning == "Yes" else "🚫"
        road_icon = "🛣️" if mainroad == "Yes" else "🚫"
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.05);border-radius:12px;padding:14px;font-size:0.85rem;color:#d1d5db'>
        📐 {area:,} sq ft &nbsp;|&nbsp; 🛏️ {bedrooms} bed &nbsp;|&nbsp; 🚿 {bathrooms} bath<br>
        🏢 {stories} floor(s) &nbsp;|&nbsp; 🚗 {parking} parking<br>
        {ac_icon} AC &nbsp;|&nbsp; {road_icon} Main Road<br>
        {furnish_short}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict House Price", use_container_width=True)

    if predict_btn:
        furnish_map = {"Furnished": 2, "Semi-Furnished": 1, "Unfurnished": 0}
        inputs = {
            "area"            : area,
            "bedrooms"        : bedrooms,
            "bathrooms"       : bathrooms,
            "stories"         : stories,
            "mainroad"        : 1 if mainroad == "Yes" else 0,
            "guestroom"       : 1 if guestroom == "Yes" else 0,
            "basement"        : 1 if basement == "Yes" else 0,
            "hotwaterheating" : 1 if hotwaterheating == "Yes" else 0,
            "airconditioning" : 1 if airconditioning == "Yes" else 0,
            "parking"         : parking,
            "prefarea"        : 1 if prefarea == "Yes" else 0,
            "furnishingstatus": furnish_map[furnishingstatus],
        }

        with st.spinner("🤖 Calculating price..."):
            price = predict_price(inputs)

        # ── Main result ──────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
            <div style="color:#9ca3af;font-size:1rem;margin-bottom:8px">PREDICTED PRICE</div>
            <div class="result-price">₹{price:,.0f}</div>
            <div class="result-sub">≈ ₹{price/1e6:.3f} Million &nbsp;|&nbsp; ₹{price/100000:.1f} Lakh</div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4 metric columns ─────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        avg_err = model_info["performance"]["MAE"]
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{price/1e6:.2f}M</div>
                <div class="metric-label">In Millions</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{price/100000:.1f}L</div>
                <div class="metric-label">In Lakhs</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">₹{price/area:,.0f}</div>
                <div class="metric-label">Per Sq Ft</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            margin = model_info["performance"]["MAPE"]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">±{margin:.1f}%</div>
                <div class="metric-label">Avg Error</div>
            </div>""", unsafe_allow_html=True)

        # ── Price range estimate ─────────────────────────────
        low  = price - avg_err
        high = price + avg_err
        st.markdown(f"""
        <div style='background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
             border-radius:12px;padding:16px;text-align:center;margin-top:16px;color:#6ee7b7'>
            <b>Estimated Range:</b>&nbsp;&nbsp;
            ₹{low:,.0f} &nbsp;–&nbsp; ₹{high:,.0f}
            <br><small style='color:#9ca3af'>Based on model's average error of ₹{avg_err:,.0f}</small>
        </div>
        """, unsafe_allow_html=True)

        # ── Compare with dataset ─────────────────────────────
        st.markdown('<div class="section-header">📊 How does this compare?</div>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 3.5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        ax.hist(df_raw['price']/1e6, bins=40,
                color='#6366f1', alpha=0.7, edgecolor='none', label='All Houses')
        ax.axvline(price/1e6, color='#34d399', linewidth=2.5,
                   linestyle='--', label=f'Your House: ₹{price/1e6:.2f}M')
        ax.axvline(df_raw['price'].mean()/1e6, color='#f59e0b', linewidth=1.5,
                   linestyle=':', label=f'Avg: ₹{df_raw["price"].mean()/1e6:.2f}M')

        ax.set_xlabel("Price (₹ Millions)", color='#9ca3af')
        ax.set_ylabel("Number of Houses", color='#9ca3af')
        ax.set_title("Your Predicted Price vs Dataset Distribution", color='#e5e7eb', fontweight='bold')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        ax.legend(facecolor='#1f2937', labelcolor='white', fontsize=9)

        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Percentile ───────────────────────────────────────
        pct = (df_raw['price'] < price).mean() * 100
        st.markdown(f"""
        <div style='text-align:center;padding:12px;background:rgba(99,102,241,0.15);
             border-radius:12px;color:#c4b5fd;margin-top:8px'>
            Your predicted house is more expensive than&nbsp;
            <b style='color:#a78bfa;font-size:1.2rem'>{pct:.1f}%</b>
            &nbsp;of houses in the dataset
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# TAB 2 — DATA EXPLORER
# ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📂 Raw Dataset</div>', unsafe_allow_html=True)

    # Top stats
    s1, s2, s3, s4, s5 = st.columns(5)
    for col_widget, label, value in [
        (s1, "Total Houses",  f"{len(df_raw)}"),
        (s2, "Features",      f"{df_raw.shape[1]-1}"),
        (s3, "Min Price",     f"₹{df_raw['price'].min()/1e6:.2f}M"),
        (s4, "Max Price",     f"₹{df_raw['price'].max()/1e6:.2f}M"),
        (s5, "Avg Price",     f"₹{df_raw['price'].mean()/1e6:.2f}M"),
    ]:
        col_widget.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rows = st.slider("Rows to show", 5, 50, 10)
    st.dataframe(df_raw.head(rows), use_container_width=True)

    st.markdown('<div class="section-header">📊 Visual Analysis</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Price distribution
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')
        ax.hist(df_raw['price']/1e6, bins=35, color='#8b5cf6',
                edgecolor='none', alpha=0.85)
        ax.set_xlabel("Price (₹ Millions)", color='#9ca3af')
        ax.set_ylabel("Count", color='#9ca3af')
        ax.set_title("Price Distribution", color='#e5e7eb', fontweight='bold')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Area vs Price scatter
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')
        sc = ax.scatter(df_raw['area'], df_raw['price']/1e6,
                        alpha=0.5, s=18,
                        c=df_raw['price'], cmap='plasma')
        ax.set_xlabel("Area (sq ft)", color='#9ca3af')
        ax.set_ylabel("Price (₹ Millions)", color='#9ca3af')
        ax.set_title("Area vs Price", color='#e5e7eb', fontweight='bold')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with chart_col2:
        # Box plots for categorical features
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')
        groups = [df_raw[df_raw['airconditioning'] == cat]['price']/1e6
                  for cat in ['no', 'yes']]
        bp = ax.boxplot(groups, patch_artist=True,
                        labels=['No AC', 'Has AC'],
                        boxprops=dict(facecolor='#6366f1', alpha=0.7),
                        medianprops=dict(color='#34d399', linewidth=2),
                        whiskerprops=dict(color='#9ca3af'),
                        capprops=dict(color='#9ca3af'),
                        flierprops=dict(markerfacecolor='#f59e0b', markersize=3))
        ax.set_ylabel("Price (₹ Millions)", color='#9ca3af')
        ax.set_title("Price: AC vs No AC", color='#e5e7eb', fontweight='bold')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Avg price by bedrooms
        avg_by_bed = df_raw.groupby('bedrooms')['price'].mean() / 1e6
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f0c29')
        ax.set_facecolor('#0f0c29')
        bars = ax.bar(avg_by_bed.index, avg_by_bed.values,
                      color=['#6366f1','#8b5cf6','#a78bfa','#c4b5fd','#e9d5ff','#f3e8ff'],
                      edgecolor='none', alpha=0.9)
        ax.set_xlabel("Number of Bedrooms", color='#9ca3af')
        ax.set_ylabel("Avg Price (₹ Millions)", color='#9ca3af')
        ax.set_title("Avg Price by Bedrooms", color='#e5e7eb', fontweight='bold')
        ax.tick_params(colors='#9ca3af')
        for spine in ax.spines.values(): spine.set_edgecolor('#374151')
        for bar, val in zip(bars, avg_by_bed.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    f'₹{val:.1f}M', ha='center', va='bottom',
                    color='#c4b5fd', fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Correlation heatmap
    st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>',
                unsafe_allow_html=True)
    num_cols = ['price','area','bedrooms','bathrooms','stories','parking']
    corr = df_raw[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#0f0c29')
    ax.set_facecolor('#0f0c29')
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdPu',
                linewidths=0.5, ax=ax,
                annot_kws={"size": 10, "color": "white"})
    ax.set_title("Feature Correlation", color='#e5e7eb', fontweight='bold')
    ax.tick_params(colors='#9ca3af')
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ───────────────────────────────────────────────────────────────
# TAB 3 — MODEL INSIGHTS
# ───────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🤖 About the Model</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);
         border-radius:16px;padding:24px;line-height:2'>
    <b style='color:#a78bfa'>Algorithm:</b> Random Forest Regressor (Hyperparameter Tuned)<br>
    <b style='color:#a78bfa'>Training:</b> 80% of 545 houses (436 samples)<br>
    <b style='color:#a78bfa'>Testing:</b> 20% of 545 houses (109 samples)<br>
    <b style='color:#a78bfa'>Validation:</b> 5-Fold Cross-Validation<br>
    <b style='color:#a78bfa'>Tuning:</b> GridSearchCV over 24 parameter combinations<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📌 Feature Importance</div>',
                unsafe_allow_html=True)

    importances = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#0f0c29')
    ax.set_facecolor('#0f0c29')

    colors = ['#6366f1' if v < importances.quantile(0.66)
              else '#8b5cf6' if v < importances.quantile(0.90)
              else '#34d399'
              for v in importances.values]

    bars = ax.barh(importances.index, importances.values,
                   color=colors, edgecolor='none', height=0.65)

    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', color='#c4b5fd', fontsize=8)

    ax.set_xlabel("Importance Score", color='#9ca3af')
    ax.set_title("Feature Importance — What drives house price?",
                 color='#e5e7eb', fontweight='bold')
    ax.tick_params(colors='#9ca3af')
    for spine in ax.spines.values(): spine.set_edgecolor('#374151')

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Top features explanation
    top5 = importances.sort_values(ascending=False).head(5)
    st.markdown('<div class="section-header">💡 Key Findings</div>',
                unsafe_allow_html=True)

    feat_explain = {
        "area"         : "Bigger area = higher price. Strongest predictor.",
        "log_area"     : "Log-transformed area — captures non-linear size effect.",
        "bathrooms"    : "More bathrooms = luxury indicator.",
        "total_rooms"  : "Total rooms (bedrooms + bathrooms) reflects house size.",
        "area_per_room": "Space per room — quality of space matters.",
        "luxury_score" : "Combined luxury: AC + hot water + preferred area + guestroom.",
        "airconditioning": "AC presence significantly raises value.",
        "stories"      : "More floors generally = larger and pricier house.",
        "bedrooms"     : "Number of bedrooms affects family suitability and price.",
        "prefarea"     : "Being in a preferred area boosts price.",
    }

    c1, c2 = st.columns(2)
    for i, (feat, val) in enumerate(top5.items()):
        col = c1 if i % 2 == 0 else c2
        explanation = feat_explain.get(feat, "Contributes to price prediction.")
        col.markdown(f"""
        <div style='background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
             border-radius:12px;padding:14px;margin-bottom:10px'>
            <div style='color:#a78bfa;font-weight:700;font-size:0.95rem'>
                #{i+1} {feat.replace("_"," ").title()} — {val*100:.1f}%
            </div>
            <div style='color:#9ca3af;font-size:0.82rem;margin-top:4px'>
                {explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model performance metrics
    st.markdown('<div class="section-header">📊 Performance Metrics Explained</div>',
                unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    metrics = [
        ("R² Score", f"{perf['R2']:.4f}",
         "61% of price variation is explained by the model. 1.0 = perfect."),
        ("MAE",      f"₹{perf['MAE']:,.0f}",
         "On average, predictions are off by this amount in rupees."),
        ("RMSE",     f"₹{perf['RMSE']:,.0f}",
         "Like MAE but penalises larger errors more heavily."),
        ("MAPE",     f"{perf['MAPE']:.2f}%",
         "On average, predictions are off by this percentage."),
    ]
    for col_w, (name, val, desc) in zip([p1, p2, p3, p4], metrics):
        col_w.markdown(f"""
        <div class="metric-card" style="height:140px">
            <div class="metric-value" style="font-size:1.3rem">{val}</div>
            <div class="metric-label">{name}</div>
            <div style='font-size:0.72rem;color:#6b7280;margin-top:8px'>{desc}</div>
        </div>""", unsafe_allow_html=True)
