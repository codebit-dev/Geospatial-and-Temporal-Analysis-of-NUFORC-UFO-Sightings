# Geospatial and Temporal Analysis of NUFORC UFO Sightings

An interactive **Streamlit** dashboard for exploring UFO sighting data from the National UFO Reporting Center (NUFORC). This project provides insights into UFO sighting trends, hotspots, and predictive modeling through a user-friendly web interface.

---

## 🔍 Features

- **Data Cleaning & Preprocessing:**  
  - Handles raw UFO sighting data with missing or inconsistent entries.  
  - Standardizes datetime, location, and shape columns for consistency.  

- **Feature Engineering:**  
  - Adds temporal (year, month, hour), spatial, and custom “eerie factor” features.  
  - Computes distance to Area 51 and classifies shape complexity.  

- **Clustering:**  
  - Applies KMeans and DBSCAN to detect geographic hotspots of UFO sightings.  
  - Visualizes clusters on interactive maps.  

- **Text Analysis:**  
  - Extracts top keywords from sighting comments using TF-IDF.  
  - Analyzes temporal trends in keyword frequency.  

- **Shape Prediction:**  
  - Trains a RandomForest classifier to predict UFO shape based on sighting attributes.  
  - Provides real-time predictions through the Streamlit interface.  

- **Interactive Visualizations:**  
  - Displays Folium maps for geographic analysis.  
  - Utilizes Plotly charts for temporal and categorical insights.  

- **User Interface:**  
  - Streamlit sidebar filters for year, shape, and cluster selection.  
  - Input fields for shape prediction and data export options.  

---

## 🚀 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/codebit-dev/Geospatial-and-Temporal-Analysis-of-NUFORC-UFO-Sightings.git
   cd Geospatial-and-Temporal-Analysis-of-NUFORC-UFO-Sightings
