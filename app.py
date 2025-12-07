# app.py

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import altair as alt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Global Power Plant Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Global Power Plant Exploratory Data Visualization Dashboard")

st.markdown(
    """
    This dashboard explores the **Global Power Plant Database** – an open-source dataset of
    power plants around the world, including capacity, fuel types, ownership and generation.  
    Use the sidebar filters to interactively explore different regions, fuel types and time periods.
    """
)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path)
    return df

DATA_PATH = "global_power_plant_database.csv"
df_raw = load_data(DATA_PATH)

# ------------------------------------------------------------
# 3. BASIC PREPROCESSING
# ------------------------------------------------------------
df = df_raw.copy()

# Keep some important columns and standardize names
# Columns from WRI spec: country, country_long, name, capacity_mw, latitude, longitude,

important_cols = [
    "country", "country_long", "name",
    "capacity_mw", "latitude", "longitude",
    "primary_fuel", "other_fuel1", "other_fuel2", "other_fuel3",
    "commissioning_year", "owner", "year_of_capacity_data",
    "generation_gwh_2013", "generation_gwh_2014", "generation_gwh_2015",
    "generation_gwh_2016", "generation_gwh_2017","generation_gwh_2018", "generation_gwh_2019",
    "estimated_generation_gwh_2013", "estimated_generation_gwh_2014",
    "estimated_generation_gwh_2015", "estimated_generation_gwh_2016",
    "estimated_generation_gwh_2017", "estimated_generation_gwh_2018", "estimated_generation_gwh_2019"
]

# Only take columns that actually exist
existing_cols = [c for c in important_cols if c in df.columns]
df = df[existing_cols]

# Safety filters for general use
# Longitude / Latitude cleaning 
df = df[
    df["latitude"].between(-90, 90) &
    df["longitude"].between(-180, 180)
]

# Capacity cleaning
df = df[df["capacity_mw"].notna()]

# Make the capacity numeric
df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")

# Discard empty and 0/negative capacity
df = df[df["capacity_mw"].notna() & (df["capacity_mw"] > 0)]

# Trim the extreme ends (1. ve 99. percentile)
q_low, q_high = df["capacity_mw"].quantile([0.01, 0.99])
df["capacity_mw_clipped"] = df["capacity_mw"].clip(lower=q_low, upper=q_high)

# Make commissioning year numeric
if "commissioning_year" in df.columns:
    df["commissioning_year"] = pd.to_numeric(df["commissioning_year"], errors="coerce")

# Create "estimated_generation_gwh" column
gen_cols = [c for c in df.columns
            if c.startswith("generation_gwh_") or c.startswith("estimated_generation_gwh_")]
if gen_cols:
    df["estimated_generation_gwh"] = df[gen_cols].mean(axis=1, skipna=True)
else:
    df["estimated_generation_gwh"] = np.nan

# Commissioning decade (e.g. 1990s, 2000s)
def decade_from_year(y):
    try:
        y = int(y)
        if y < 1900 or y > 2050:
            return "Unknown"
        base = (y // 10) * 10
        return f"{base}s"
    except Exception:
        return "Unknown"

if "commissioning_year" in df.columns:
    df["commissioning_decade"] = df["commissioning_year"].apply(decade_from_year)
else:
    df["commissioning_decade"] = "Unknown"
    
# Decade order
decade_order = sorted(df["commissioning_decade"].unique())
df["commissioning_decade"] = pd.Categorical(
    df["commissioning_decade"],
    categories=[d for d in decade_order if d != "Unknown"] + ["Unknown"],
    ordered=True
)

# Fill in the blank country names
if "country_long" in df.columns:
    df["country_long"] = df["country_long"].fillna(df["country"])
else:
    df["country_long"] = df["country"]

# "Unknown" if primary_fuel is empty
df["primary_fuel"] = df["primary_fuel"].fillna("Unknown")

# --- Dataset-specific derived additional feature ---
# Renewable / fossil flag
renewables = {
    "Hydro", "Wind", "Solar", "Biomass",
    "Geothermal", "Wave and Tidal", "Storage"
}
df["is_renewable"] = df["primary_fuel"].isin(renewables)


# ------------------------------------------------------------
# 4. SIDEBAR FILTERS (GLOBAL INTERACTIVITY)
# ------------------------------------------------------------
st.sidebar.header("Filters")

# Fuel type filter
fuel_types = sorted(df["primary_fuel"].dropna().unique().tolist())
fuel_options = ["All"] + fuel_types

selected_fuels = st.sidebar.multiselect(
    "Primary Fuel Types",
    options=fuel_options,
    default=["All"]
)
# Country filter "
top_country_capacity = (
    df.groupby("country_long")["capacity_mw"]
    .sum()
    .sort_values(ascending=False)
)
top_countries = top_country_capacity.head(30).index.tolist()
country_options = ["All"] + top_countries

selected_countries = st.sidebar.multiselect(
    "Select Countries (top by capacity)",
    options=country_options,
    default=["All"]
)
# Capacity range filter
cap_min = float(df["capacity_mw"].min())
cap_max = float(df["capacity_mw"].max())

capacity_range = st.sidebar.slider(
    "Installed Capacity (MW)",
    min_value=int(cap_min),
    max_value=int(cap_max),
    value=(int(cap_min), int(cap_max))
)

# Year range filter (commissioning_year)
if "commissioning_year" in df.columns and df["commissioning_year"].notna().sum() > 0:
    year_min = int(df["commissioning_year"].min())
    year_max = int(df["commissioning_year"].max())
    year_range = st.sidebar.slider(
        "Commissioning Year",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )
else:
    year_range = None

# Apply filters
filtered_df = df.copy()

# Country filter
if "All" not in selected_countries:
    filtered_df = filtered_df[filtered_df["country_long"].isin(selected_countries)]

# Fuel filter
if "All" not in selected_fuels:
    filtered_df = filtered_df[filtered_df["primary_fuel"].isin(selected_fuels)]

# Capacity filter
filtered_df = filtered_df[
    (filtered_df["capacity_mw"] >= capacity_range[0]) &
    (filtered_df["capacity_mw"] <= capacity_range[1])
]

# Year filter
if year_range is not None and "commissioning_year" in filtered_df.columns:
    y0, y1 = year_range

    # If the slider is in full range (the user did not narrow down the year)
    # -> Also keep the plants with commissioning year NaN
    if y0 == year_min and y1 == year_max:
        filtered_df = filtered_df[
            filtered_df["commissioning_year"].isna() |
            filtered_df["commissioning_year"].between(y0, y1, inclusive="both")
        ]
    else:
        # If the user narrowed down the year range
        # -> Exclude NaN years, only show those in the selected range
        filtered_df = filtered_df[
            filtered_df["commissioning_year"].between(y0, y1, inclusive="both")
        ]

st.sidebar.markdown(f"**Total plants after filters:** {len(filtered_df):,}")

# ------------------------------------------------------------
# 5. TOP METRICS
# ------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_capacity = filtered_df["capacity_mw"].sum()
avg_capacity = filtered_df["capacity_mw"].mean()
num_countries = filtered_df["country_long"].nunique()
num_fuels = filtered_df["primary_fuel"].nunique()

col1.metric("Total Plants", f"{len(filtered_df):,}")
col2.metric("Total Installed Capacity (MW)", f"{total_capacity:,.0f}")
col3.metric("Average Plant Capacity (MW)", f"{avg_capacity:,.1f}")
col4.metric("Countries in Filter", f"{num_countries} ({num_fuels} fuel types)")

st.markdown("---")

# ------------------------------------------------------------
# 6. TABS FOR VISUALIZATIONS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Overview & Basic Charts",
    "Advanced Visualizations",
    "Clustering & ML"
])

# ------------------------------------------------------------
# TAB 1 – OVERVIEW
# ------------------------------------------------------------
with tab1:
    st.subheader("1. Top Countries by Total Installed Capacity")

    # A small control panel
    col_bar1, col_bar2 = st.columns(2)
    with col_bar1:
        top_n = st.slider(
            "Number of countries to display",
            min_value=5,
            max_value=25,
            value=10,
            step=1,
            key="top_countries_slider"
        )
    with col_bar2:
       # Plant type filter
        plant_type_filter = st.selectbox(
            "Plant type filter",
            ["All plants", "Only renewables", "Only non-renewables"]
        )

    # Basic data
    bar_df = filtered_df.copy()

    # Optional renewable filter
    if "is_renewable" in bar_df.columns:
        if plant_type_filter == "Only renewables":
            bar_df = bar_df[bar_df["is_renewable"]]
        elif plant_type_filter == "Only non-renewables":
            bar_df = bar_df[~bar_df["is_renewable"]]

    # Total capacity by country
    country_cap = (
        bar_df
        .groupby("country_long", as_index=False)["capacity_mw"]
        .sum()
        .sort_values("capacity_mw", ascending=False)
        .head(top_n)
    )

    if country_cap.empty:
        st.info("No data available for the selected filters.")
    else:
        fig_bar = px.bar(
            country_cap,
            x="country_long",
            y="capacity_mw",
            title="Top Countries by Total Installed Capacity",
            labels={
                "country_long": "Country",
                "capacity_mw": "Total Installed Capacity (MW)",
            }
        )
        fig_bar.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown(
        """
        **Insight:** This chart shows which countries have the highest total installed capacity
        under the current filters.
        """
    )


    st.subheader("2. Capacity Distribution by Primary Fuel (Boxplot)")

    fig_box = px.box(
        filtered_df,
        x="primary_fuel",
        y="capacity_mw_clipped",
        color="primary_fuel",
        title="Capacity Distribution by Primary Fuel (Clipped)",
        labels={"primary_fuel": "Primary Fuel", "capacity_mw_clipped": "Capacity (MW, clipped)"}
    )
    fig_box.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown(
        """
        **Insight:** Different fuel types have very different typical plant sizes.
        For example, nuclear and large hydro plants tend to have much higher capacity than small solar or wind plants.
        """
    )

    st.subheader("3. Capacity vs. Estimated Generation (Scatter Plot)")

    if filtered_df["estimated_generation_gwh"].notna().sum() > 0:
        scatter_sample = filtered_df[
            filtered_df["estimated_generation_gwh"].notna()
        ].copy()
        if len(scatter_sample) > 5000:
            scatter_sample = scatter_sample.sample(5000, random_state=42)

        fig_scatter = px.scatter(
            scatter_sample,
            x="capacity_mw",
            y="estimated_generation_gwh",
            color="primary_fuel",
            hover_data=["name", "country_long"],
            title="Capacity (MW) vs Estimated Annual Generation (GWh)",
            labels={
                "capacity_mw": "Installed Capacity (MW)",
                "estimated_generation_gwh": "Estimated Generation (GWh)"
            }
        )
        fig_scatter.update_layout(xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Estimated generation data not available in this version of the dataset.")

# ------------------------------------------------------------
# TAB 2 – ADVANCED VISUALIZATIONS
# ------------------------------------------------------------
with tab2:
    st.subheader("4. Global Map of Power Plants")

    st.markdown(
        " Map of power plants colored by primary fuel and sized by capacity."
    )

    # Renewable filter + color mode for map
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        renewable_filter_map = st.radio(
            "Filter plants on map",
            ["All plants", "Only renewables", "Only non-renewables"],
            horizontal=False
        )
    with col_map2:
        color_mode_map = st.radio(
            "Color points by",
            ["Primary fuel", "Renewable vs Non-renewable"],
            horizontal=False
        )

    map_sample = filtered_df.copy()

    if renewable_filter_map == "Only renewables":
        map_sample = map_sample[map_sample["is_renewable"]]
    elif renewable_filter_map == "Only non-renewables":
        map_sample = map_sample[~map_sample["is_renewable"]]

    # Performance sampling for the map
    if len(map_sample) > 10000:
        map_sample = map_sample.sample(10000, random_state=42)

    # Color column
    if color_mode_map == "Primary fuel":
        color_col = "primary_fuel"
    else:
        map_sample["renewable_label"] = np.where(
            map_sample["is_renewable"], "Renewable", "Non-renewable"
        )
        color_col = "renewable_label"
        

    fig_map = px.scatter_mapbox(
        map_sample,
        lat="latitude",
        lon="longitude",
        color=color_col,
        size="capacity_mw_clipped",
        hover_name="name",
        hover_data=["country_long", "primary_fuel", "capacity_mw", "estimated_generation_gwh"],
        zoom=1,
        height=600,
        title="Global Distribution of Power Plants"
    )
    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("5. Treemap – Capacity by Renewables, Fuel and Country")

    treemap_mode = st.radio(
        "Treemap plant type",
        ["All plants", "Only renewables", "Only non-renewables"],
        horizontal=True
    )

    treemap_base = filtered_df.copy()
    if treemap_mode == "Only renewables":
        treemap_base = treemap_base[treemap_base["is_renewable"]]
    elif treemap_mode == "Only non-renewables":
        treemap_base = treemap_base[~treemap_base["is_renewable"]]

    treemap_base["renewable_label"] = np.where(
        treemap_base["is_renewable"], "Renewable", "Non-renewable"
    )

    treemap_df = (
        treemap_base
        .groupby(["renewable_label", "primary_fuel", "country_long"])["capacity_mw"]
        .sum()
        .reset_index()
        .rename(columns={"capacity_mw": "total_capacity_mw"})
    )

    fig_treemap = px.treemap(
        treemap_df,
        path=["renewable_label", "primary_fuel", "country_long"],
        values="total_capacity_mw",
        color="renewable_label",
        title="Treemap of Total Installed Capacity by Renewables, Fuel and Country",
    )
    st.plotly_chart(fig_treemap, use_container_width=True)


    st.subheader("6. Sunburst – Fuel Mix by Commissioning Decade")

    sunburst_df = (
        filtered_df
        .groupby(["primary_fuel", "commissioning_decade"])["capacity_mw"]
        .sum()
        .reset_index()
        .rename(columns={"capacity_mw": "total_capacity_mw"})
    )

    fig_sunburst = px.sunburst(
        sunburst_df,
        path=["primary_fuel", "commissioning_decade"],
        values="total_capacity_mw",
        title="Sunburst – Capacity by Fuel Type and Commissioning Decade"
    )
    st.plotly_chart(fig_sunburst, use_container_width=True)

    st.subheader("7. Parallel Coordinates – Capacity, Year, Generation")

    par_cols = ["capacity_mw_clipped", "commissioning_year", "estimated_generation_gwh"]
    par_df = filtered_df[par_cols + ["primary_fuel"]].dropna()

    if len(par_df) > 2000:
        par_df = par_df.sample(2000, random_state=42)

    if not par_df.empty:
        fig_parallel = px.parallel_coordinates(
            par_df,
            dimensions=par_cols,
            color="capacity_mw_clipped",
            color_continuous_scale=px.colors.diverging.Tealrose,
            labels={
                "capacity_mw_clipped": "Capacity (MW, clipped)",
                "commissioning_year": "Year",
                "estimated_generation_gwh": "Est. Generation (GWh)"
            },
            title="Parallel Coordinates of Key Numerical Features"
        )
        st.plotly_chart(fig_parallel, use_container_width=True)
    else:
        st.info("Not enough complete data for parallel coordinates plot.")

    st.subheader("8. Heatmap – Fuel Type vs. Commissioning Decade (Total Capacity)")

    heat_df = (
        filtered_df
        .groupby(["primary_fuel", "commissioning_decade"])["capacity_mw"]
        .sum()
        .reset_index()
        .rename(columns={"capacity_mw": "total_capacity_mw"})
    )

    heat_chart = alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("commissioning_decade:N", title="Commissioning Decade"),
        y=alt.Y("primary_fuel:N", title="Primary Fuel"),
        color=alt.Color("total_capacity_mw:Q", title="Total Capacity (MW)"),
        tooltip=["primary_fuel", "commissioning_decade", "total_capacity_mw"]
    ).properties(
        width=650,
        height=400,
        title="Heatmap – Total Installed Capacity by Fuel and Decade"
    )

    st.altair_chart(heat_chart, use_container_width=True)

    st.subheader("9. Sankey Diagram – Fuel → Country Flows (by Capacity)")

    # Top countries for Sankey
    sankey_df = (
        filtered_df
        .groupby(["primary_fuel", "country_long"])["capacity_mw"]
        .sum()
        .reset_index()
        .rename(columns={"capacity_mw": "total_capacity_mw"})
    )

    # Only the top 10 countries with the highest capacity
    top_sankey_countries = (
        sankey_df.groupby("country_long")["total_capacity_mw"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    sankey_df["country_node"] = np.where(
        sankey_df["country_long"].isin(top_sankey_countries),
        sankey_df["country_long"],
        "Other"
    )

    sankey_agg = (
        sankey_df
        .groupby(["primary_fuel", "country_node"])["total_capacity_mw"]
        .sum()
        .reset_index()
    )

    # Node index mapping
    fuel_nodes = sankey_agg["primary_fuel"].unique().tolist()
    country_nodes = sankey_agg["country_node"].unique().tolist()

    all_nodes = fuel_nodes + country_nodes
    node_index = {name: i for i, name in enumerate(all_nodes)}

    sources = sankey_agg["primary_fuel"].map(node_index)
    targets = sankey_agg["country_node"].map(node_index)
    values = sankey_agg["total_capacity_mw"]

    link = dict(
        source=sources,
        target=targets,
        value=values
    )

    node = dict(
        pad=15,
        thickness=20,
        label=all_nodes
    )

    fig_sankey = go.Figure(data=[go.Sankey(node=node, link=link)])
    fig_sankey.update_layout(
        title_text="Sankey – Capacity Flow from Fuel Types to Countries",
        font=dict(size=12)
    )

    st.plotly_chart(fig_sankey, use_container_width=True)

# ------------------------------------------------------------
# TAB 3 – SIMPLE ML: K-MEANS CLUSTERING
# ------------------------------------------------------------
with tab3:
    st.subheader("10. K-Means Clustering of Power Plants")

    st.markdown(
        """
        We apply a simple **K-Means** clustering on two numerical features:
        - Installed capacity (MW)
        - Estimated generation (GWh)  
        
        This groups power plants into segments such as *small–low generation*, *medium*, and *large–high generation*.
        """
    )

    if filtered_df["estimated_generation_gwh"].notna().sum() > 0:
        cluster_data = filtered_df[["capacity_mw_clipped", "estimated_generation_gwh"]].dropna().copy()

        if len(cluster_data) > 5000:
            cluster_data = cluster_data.sample(5000, random_state=42)

        # Slider for number of clusters
        k = st.slider("Number of clusters (k)", 2, 8, 4, key="cluster_k")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_data)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(X_scaled)

        cluster_data["cluster"] = cluster_labels

        fig_cluster = px.scatter(
            cluster_data,
            x="capacity_mw_clipped",
            y="estimated_generation_gwh",
            color="cluster",
            title=f"K-Means Clusters (k={k}) on Capacity and Estimated Generation",
            labels={
                "capacity_mw_clipped": "Capacity (MW, clipped)",
                "estimated_generation_gwh": "Est. Generation (GWh)"
            },
            opacity=0.8
        )
        fig_cluster.update_layout(xaxis_type="log", yaxis_type="log")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("### Cluster Summary (mean values)")
        st.write(cluster_data.groupby("cluster")[["capacity_mw_clipped", "estimated_generation_gwh"]].mean())
    else:
        st.info("Estimated generation column not available in this dataset version. Clustering demo is disabled.")
