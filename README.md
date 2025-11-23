# Global Power Plant Exploratory Data Visualization Dashboard

This project is an interactive dashboard built with **Python** and **Streamlit** to explore the **Global Power Plant Database**. The dataset contains grid-scale power plants (≥ 1 MW) around the world with information on location, capacity, fuel type, ownership and generation. :contentReference[oaicite:2]{index=2}

## 1. Dataset

- **Name:** Global Power Plant Database (v1.3)
- **Source:** World Resources Institute (WRI) / Kaggle
- **Rows:** ≈ 30,000+ power plants from 160+ countries
- **Columns (subset used):**
  - `country`, `country_long`
  - `name`
  - `latitude`, `longitude`
  - `capacity_mw`
  - `primary_fuel`, `other_fuel1–3`
  - `commissioning_year`, `year_of_capacity_data`
  - `generation_gwh_2013–2017`
  - `estimated_generation_gwh_2013–2017`
- Each row represents a single power plant. :contentReference[oaicite:3]{index=3}  

Dataset link: *(add the exact Kaggle / WRI URL you used)*

## 2. Project Goals

- Clean and preprocess the dataset (handle missing values, clip outliers, derive new features).
- Build an interactive dashboard that:
  - Shows global patterns of installed capacity and fuel mix.
  - Visualizes spatial distribution of power plants.
  - Explores relationships between capacity, commissioning year and generation.
  - Uses at least **9 distinct visualizations**, including at least **6 advanced** charts.
  - Demonstrates basic machine learning (K-Means clustering).

## 3. Visualization Techniques

The dashboard implements the following visualizations:

1. **Bar Chart** – Top countries by total installed capacity (MW).
2. **Boxplot** – Distribution of plant capacity by primary fuel type.
3. **Scatter Plot** – Capacity vs. estimated annual generation (log–log scale).
4. **World Map (Mapbox)** – Geographic distribution of plants, colored by fuel and sized by capacity.
5. **Treemap** – Total capacity by fuel type and country.
6. **Sunburst Diagram** – Capacity by fuel type and commissioning decade.
7. **Parallel Coordinates Plot** – Capacity, commissioning year and estimated generation.
8. **Heatmap (Altair)** – Total capacity by fuel type and commissioning decade.
9. **K-Means Clustering Scatter Plot** – Clusters of plants based on capacity and generation.

All charts are interactive (hover, zoom, pan, filter). The sidebar includes:
- Country filter (multiselect)
- Primary fuel filter (multiselect)
- Capacity range slider
- Commissioning year range slider
- Cluster number slider (for K-Means)

## 4. Installation & Usage

```bash
# clone the repo
git clone https://github.com/<your-username>/global-power-plant-dashboard.git
cd global-power-plant-dashboard

# (optional) create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# install dependencies
pip install -r requirements.txt

# make sure the dataset is in the data/ folder
# e.g. data/global_power_plant_database.csv

# run the app
streamlit run app.py
