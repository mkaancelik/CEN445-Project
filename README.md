# Global Power Plant Exploratory Data Visualization Dashboard

This repository contains an interactive data visualization dashboard built with **Python** and **Streamlit** to explore the **Global Power Plant Database**. The application allows users to analyze power plants around the world by fuel type, location, capacity, and commissioning year, and includes both advanced visualizations and a simple machine learning component (K-Means clustering).

---

## 1. Project Description

The main objective of this project is to design and implement an **exploratory data visualization dashboard** that:

- Provides an interactive and user-friendly interface for exploring the global power plant dataset.
- Uses multiple advanced visualization techniques (treemap, sunburst, parallel coordinates, Sankey diagram, maps, etc.).
- Enables users to filter and drill down into the data using controls such as dropdowns, sliders, and multiselect widgets.
- Demonstrates the use of a basic machine learning model (K-Means) to cluster power plants based on their characteristics.

The dashboard is organized into three main tabs:

1. **Overview & Basic Charts** – high-level country and fuel comparisons.
2. **Advanced Visualizations** – spatial, hierarchical, and multivariate views.
3. **Clustering & ML** – K-Means clustering on selected numerical features.

---

## 2. Dataset Details

**Dataset name:** Global Power Plant Database (v1.3)  
**Source:** World Resources Institute (WRI) / Kaggle (open data)  

Each row represents a single grid-scale power plant (typically ≥ 1 MW) and includes:

- **Identification & location**
  - `name` – Plant name  
  - `country`, `country_long` – Country code and full name  
  - `latitude`, `longitude` – Geographic coordinates  

- **Technical attributes**
  - `capacity_mw` – Installed capacity in megawatts (MW)  
  - `primary_fuel` – Main fuel type (Coal, Gas, Hydro, Wind, Solar, Nuclear, Biomass, etc.)  
  - `other_fuel1`, `other_fuel2`, `other_fuel3` – Additional fuels (if any)  

- **Temporal attributes**
  - `commissioning_year` – Approximate year the plant was commissioned  
  - `year_of_capacity_data` – Year of capacity data (if available)  

- **Generation-related attributes**
  - `generation_gwh_20xx` – Reported generation in specific years (if available)  
  - `estimated_generation_gwh_20xx` – Estimated annual generation for some years  

In our project, we also derive:

- `capacity_mw_clipped` – Capacity clipped to the 1st and 99th percentile to reduce extreme outliers.
- `commissioning_decade` – Decade label (e.g. *1990s*, *2000s*).
- `estimated_generation_gwh` – Average across available generation-related columns.

### Dataset File / Source Link

- Local file used by the app:  
  `data/global_power_plant_database.csv`

- Example source link (replace with the exact URL you used):  
  - WRI GitHub: `<https://github.com/wri/global-power-plant-database>`  
  - or Kaggle: `<https://www.kaggle.com/>` (search: “Global Power Plant Database”)

---

## 3. Features and Visualizations

The dashboard includes at least **9 distinct visualizations**, with **6+ advanced types**, all interactive:

1. **Bar chart** – Top countries by total installed capacity (MW).  
2. **Boxplot** – Distribution of plant capacity by primary fuel type.  
3. **Scatter plot (log–log)** – Installed capacity vs estimated generation.  
4. **Global map (Mapbox)** – Locations of plants, colored by fuel type, sized by capacity.  
5. **Treemap** – Total capacity by primary fuel and country.  
6. **Sunburst diagram** – Capacity by primary fuel and commissioning decade.  
7. **Parallel coordinates plot** – Capacity (clipped), commissioning year, estimated generation.  
8. **Heatmap (Altair)** – Total capacity by fuel type and commissioning decade.  
9. **Sankey diagram** – Capacity flow from fuel types to countries (top 10 + “Other”).  
10. **K-Means clustering scatter plot** – Clusters of plants based on capacity and estimated generation.

**Interactivity includes:**

- Sidebar filters (Streamlit):
  - Country multiselect  
  - Primary fuel multiselect  
  - Capacity range slider  
  - Commissioning year range slider  
  - Cluster number slider (for K-Means)  
- Interactive Plotly and Altair charts with:
  - Hover tooltips  
  - Zooming and panning  
  - Dynamic filtering based on sidebar selections  

---

4.2. Create and Activate a Virtual Environment
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
# source venv/bin/activate
4.3. Install Dependencies

Make sure a requirements.txt file exists with at least:
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
# source venv/bin/activate
Then run:
pip install -r requirements.txt
4.4. Prepare the Dataset

Create a folder named data in the project root if it does not exist.

Download the Global Power Plant Database CSV file.

Save it as:data/global_power_plant_database.csv
4.5. Run the Streamlit Application
streamlit run app.py
Open the URL shown in the terminal (usually http://localhost:8501) in your browser.
5. Repository Structure
global-power-plant-dashboard/
│
├─ app.py                       # Main Streamlit application
├─ requirements.txt             # Python dependencies
├─ README.md                    # Project documentation (this file)
├─ report.pdf / report.md       # One-page project report (optional)
└─ data/
   └─ global_power_plant_database.csv   # Dataset file (not committed if large)

