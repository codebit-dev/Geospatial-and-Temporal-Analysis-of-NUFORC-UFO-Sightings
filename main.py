# ==========================================================
# 🛸 NUFORC UFO Explorer - Streamlit Dashboard
# ==========================================================
import streamlit as st
st.set_page_config(layout="wide", page_title="NUFORC UFO Explorer")

# --- Imports ---
import pandas as pd
import numpy as np
import re
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
import plotly.express as px
import seaborn as sns
import os

sns.set_style("darkgrid")

# ==========================================================
# -------- DATA CLEANING (Person A) --------
# ==========================================================
@st.cache_data(show_spinner=False)
def load_data(path="complete.csv"):
    df = pd.read_csv(path, low_memory=False, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]
    return df

def clean_text(s):
    if pd.isna(s):
        return ""
    s = re.sub(r'&#\d+;|&amp;|&quot;|&apos;|&nbsp;', ' ', str(s))
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

@st.cache_data(show_spinner=False)
def preprocess(df):
    df = df.copy()
    df['comments'] = df.get('comments', "").apply(clean_text) if 'comments' in df.columns else ""
    df['shape'] = df.get('shape', pd.Series(["unknown"]*len(df))).fillna('unknown').astype(str).str.lower()
    # Detect lat/lon columns
    lat_col = None; lon_col = None
    for c in df.columns:
        if c.strip().lower() == 'latitude':
            lat_col = c
        if c.strip().lower().startswith('long'):
            lon_col = c
    if lat_col is None or lon_col is None:
        raise KeyError("Latitude/Longitude columns not found.")
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
    # Duration & datetime
    df['duration (seconds)'] = pd.to_numeric(df.get('duration (seconds)', pd.Series([np.nan]*len(df))), errors='coerce')
    df['duration_min'] = df['duration (seconds)'] / 60.0
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['year']  = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day']   = df['datetime'].dt.day
    df['hour']  = df['datetime'].dt.hour.fillna(0).astype(int)
    df = df.dropna(subset=['datetime','latitude','longitude']).reset_index(drop=True)
    df['duration_min'] = df['duration_min'].clip(upper=df['duration_min'].quantile(0.99))
    return df

# ==========================================================
# -------- FEATURE ENGINEERING (Person B) --------
# ==========================================================
def get_season(m):
    if pd.isna(m): return 'unknown'
    m = int(m)
    if m in [12,1,2]: return 'Winter'
    if m in [3,4,5]: return 'Spring'
    if m in [6,7,8]: return 'Summer'
    return 'Fall'

@st.cache_data(show_spinner=False)
def engineer_features(df):
    df = df.copy()
    df['night_flag'] = df['hour'].apply(lambda h: 1 if (h >= 20 or h <= 5) else 0)
    df['season'] = df['month'].apply(get_season)
    simple_shapes = {'circle','disk','light','sphere','oval','round'}
    medium_shapes = {'triangle','cigar','cone','delta'}
    def sc(s):
        if s in simple_shapes: return 1
        if s in medium_shapes: return 2
        return 3
    df['shape_complexity'] = df['shape'].apply(sc)
    df['lights_mentioned'] = df['comments'].str.contains(r'\blight|\bglow|\bbright\b', case=False, na=False).astype(int)
    def hav(lat1, lon1, lat2=37.24, lon2=-115.81):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2-lat1, lon2-lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))
    df['dist_area51_km'] = df.apply(lambda r: hav(r['latitude'], r['longitude']), axis=1)
    keywords = ['abduct','beam','alien','probe','strange','weird','unidentified']
    def eerie(r):
        c = r['comments'] if pd.notna(r['comments']) else ""
        score = sum(1 for kw in keywords if kw in c)
        score += (r['duration_min'] or 0) / 10.0
        return round(score,2)
    df['eerie_factor'] = df.apply(eerie, axis=1)
    return df

# ==========================================================
# -------- CLUSTERING & HOTSPOTS (Person C) --------
# ==========================================================
@st.cache_data(show_spinner=False)
def run_clustering(df, k=8, db_eps=0.5, db_min_samples=15):
    geo = df[['latitude','longitude']].dropna()
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(geo)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(geo)
    db = DBSCAN(eps=db_eps, min_samples=db_min_samples)
    db_labels = db.fit_predict(coords_scaled)
    cluster_df = geo.copy()
    cluster_df['kmeans_cluster'] = km_labels
    cluster_df['dbscan_cluster'] = db_labels
    df = df.copy()
    df.loc[cluster_df.index, 'kmeans_cluster'] = cluster_df['kmeans_cluster']
    df.loc[cluster_df.index, 'dbscan_cluster'] = cluster_df['dbscan_cluster']
    return df, km, db, scaler

# ==========================================================
# -------- TEXT ANALYSIS & VISUALIZATION (Person D) --------
# ==========================================================
@st.cache_data(show_spinner=False)
def compute_tfidf_keywords(texts, top_k=30):
    tfidf = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1,2))
    X = tfidf.fit_transform(texts.fillna(""))
    avg = X.mean(axis=0).A1
    features = np.array(tfidf.get_feature_names_out())
    order = avg.argsort()[::-1]
    top_terms = features[order][:top_k]
    top_scores = avg[order][:top_k]
    return pd.DataFrame({'term': top_terms, 'score': top_scores})

# ==========================================================
# -------- SHAPE PREDICTION MODEL (Person E - YOU) --------
# ==========================================================
@st.cache_resource
def train_shape_model(df, top_n=10):
    top_shapes = df['shape'].value_counts().head(top_n).index.tolist()
    clf_df = df[df['shape'].isin(top_shapes)].dropna(subset=['latitude','longitude','hour','duration_min'])
    if clf_df.shape[0] < 100:
        return None, None, None
    le = LabelEncoder()
    y = le.fit_transform(clf_df['shape'])
    X = clf_df[['latitude','longitude','year','month','hour','duration_min','night_flag','shape_complexity']].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    return rf, le, acc

# ==========================================================
# -------- STREAMLIT APP UI --------
# ==========================================================
st.title("🛸 NUFORC UFO Explorer")
st.markdown("Interactive dashboard: hotspots, trends, keywords & shape prediction")

# Sidebar controls
st.sidebar.header("Configuration")
data_path = st.sidebar.text_input("CSV path", value="complete.csv")
k_clusters = st.sidebar.slider("KMeans clusters (k)", 3, 20, 8)
db_eps = st.sidebar.slider("DBSCAN eps (scaled)", 0.1, 2.0, 0.5, step=0.05)
db_min = st.sidebar.slider("DBSCAN min samples", 3, 50, 15)
top_keywords = st.sidebar.slider("Top TF-IDF keywords", 5, 50, 20)
top_shapes = st.sidebar.slider("Top shapes (for model)", 5, 20, 10)
sample_map = st.sidebar.slider("Map sample size", 500, 5000, 2000, step=500)

# Load and preprocess data
with st.spinner("Loading data..."):
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Upload to working folder or change path.")
        st.stop()
    raw = load_data(data_path)
    df = preprocess(raw)
    df = engineer_features(df)

st.sidebar.success("Data loaded")

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", f"{len(df):,}")
c2.metric("Year range", f"{int(df['year'].min())} - {int(df['year'].max())}")
c3.metric("Unique shapes", df['shape'].nunique())
c4.metric("Top country", df['country'].mode().iat[0] if 'country' in df.columns else "N/A")

# Filters
st.subheader("Filters")
col1, col2, col3 = st.columns([3,2,2])
years = sorted(df['year'].dropna().unique())
selected_years = col1.multiselect("Year(s)", options=years, default=[years[-1]] if years else [])
shapes = sorted(df['shape'].value_counts().index.tolist())
selected_shapes = col2.multiselect("Shape(s)", options=shapes[:100], default=shapes[:5])
cluster_choice = col3.selectbox("Color clusters by", options=["None","kmeans_cluster","dbscan_cluster"])

# Apply filters
filtered = df.copy()
if selected_years:
    filtered = filtered[filtered['year'].isin(selected_years)]
if selected_shapes:
    filtered = filtered[filtered['shape'].isin(selected_shapes)]

st.write(f"Showing {len(filtered):,} records after filtering")

# Clustering
with st.spinner("Clustering..."):
    filtered_with_clusters, km_model, db_model, scaler = run_clustering(filtered, k=k_clusters, db_eps=db_eps, db_min_samples=db_min)

# Map
st.subheader("Hotspot Map (Folium)")
map_center = [filtered_with_clusters['latitude'].mean(), filtered_with_clusters['longitude'].mean()]
m = folium.Map(
    location=map_center,
    zoom_start=3,
    tiles="Stamen Toner Lite",
    attr='Map tiles by Stamen Design, under CC BY 3.0 — Map data © OpenStreetMap contributors'
)


# Heatmap
heat_data = filtered_with_clusters[['latitude','longitude']].dropna().sample(min(sample_map, len(filtered_with_clusters))).values.tolist()
HeatMap(heat_data, radius=6, blur=10).add_to(m)

# KMeans centers
if km_model is not None:
    for idx, center in enumerate(km_model.cluster_centers_):
        folium.CircleMarker(location=[center[0], center[1]], radius=6, color='red', fill=True,
                            popup=f"KMeans center {idx}").add_to(m)

# Points with cluster color
points = filtered_with_clusters.sample(min(sample_map, len(filtered_with_clusters)), random_state=42)
for _, r in points.iterrows():
    color = "blue"
    if cluster_choice == "kmeans_cluster" and pd.notna(r.get('kmeans_cluster')):
        color = f"#{(int(r['kmeans_cluster'])*37)%255:02x}55{(int(r['kmeans_cluster'])*17)%255:02x}"
    if cluster_choice == "dbscan_cluster" and pd.notna(r.get('dbscan_cluster')):
        color = 'gray' if r['dbscan_cluster']==-1 else f"#{(int(r['dbscan_cluster'])*97)%255:02x}33{(int(r['dbscan_cluster'])*23)%255:02x}"
    folium.CircleMarker(location=[r['latitude'], r['longitude']],
                        radius=2, color=color, fill=True, fill_opacity=0.6,
                        popup=f"{r.get('datetime')}<br>{r.get('city','')}<br>{r.get('shape','')}").add_to(m)

map_html = m._repr_html_()
components.html(map_html, height=900, width=1000)

# Temporal plots
st.subheader("Temporal Trends")
colA, colB = st.columns(2)
with colA:
    yearly = df.groupby('year').size().reset_index(name='count').dropna()
    fig1 = px.line(yearly, x='year', y='count', title="Sightings per Year")
    st.plotly_chart(fig1, use_container_width=True)
with colB:
    hourly = df.groupby('hour').size().reset_index(name='count')
    fig2 = px.bar(hourly, x='hour', y='count', title="Sightings by Hour")
    st.plotly_chart(fig2, use_container_width=True)

# TF-IDF keywords
st.subheader("Top TF-IDF Keywords in Comments")
with st.spinner("Computing TF-IDF..."):
    tfidf_df = compute_tfidf_keywords(df['comments'], top_k=top_keywords)
st.table(tfidf_df.head(top_keywords))

kw_fig = px.bar(tfidf_df.head(top_keywords), x='score', y='term', orientation='h', title="Top Keywords")
st.plotly_chart(kw_fig, use_container_width=True)

# Shape prediction tool
st.subheader("Predict UFO Shape (RandomForest baseline)")
with st.spinner("Training shape model..."):
    rf_model, label_enc, model_acc = train_shape_model(df, top_n=top_shapes)
if rf_model is None:
    st.info("Not enough data to train shape model on top shapes.")
else:
    st.text(f"Trained RandomForest on top {top_shapes} shapes — validation accuracy: {model_acc:.3f}")
    st.markdown("Enter values to predict shape:")
    pcol1, pcol2, pcol3 = st.columns(3)
    lat_in = pcol1.number_input("Latitude", value=float(df['latitude'].mean()))
    lon_in = pcol1.number_input("Longitude", value=float(df['longitude'].mean()))
    year_in = pcol2.number_input("Year", value=int(df['year'].median()))
    month_in = pcol2.slider("Month", 1, 12, 7)
    hour_in = pcol3.slider("Hour", 0, 23, 21)
    duration_in = pcol3.number_input("Duration (min)", value=float(df['duration_min'].median()))
    if st.button("Predict shape"):
        Xpred = np.array([[lat_in, lon_in, year_in, month_in, hour_in, duration_in, 1 if (hour_in>=20 or hour_in<=5) else 0, 2]])
        pred = rf_model.predict(Xpred)
        shape_pred = label_enc.inverse_transform(pred)[0]
        st.success(f"Predicted shape: {shape_pred}")

# Save processed CSV
st.write("---")
if st.button("Save processed CSV (ufo_processed.csv)"):
    df.to_csv("ufo_processed.csv", index=False)
    st.success("Saved processed CSV → ufo_processed.csv")

# Footer
st.markdown("Built by: Avinaba Roy, Chetona Roy, Devdeep Hazra, Diya Hazra, Somnath Chakraborty")



