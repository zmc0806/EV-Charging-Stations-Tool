import streamlit as st
import folium
import branca.colormap as cm
import pandas as pd
import geopandas as gpd
import numpy as np
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
from shapely.geometry import Point, Polygon, MultiPolygon
from folium.plugins import Draw, MousePosition, MeasureControl
from sklearn.decomposition import PCA

from weight_optimization import WeightOptimizer_1
from weight_analysis_ui import render_weight_analysis_ui_1
from weight_impact_analysis import WeightImpactAnalyzer


# ----- Corrected PCA Weight Optimizer -----
class WeightOptimizer:
    @staticmethod
    def optimize_weights(df):
        # Extract feature columns ending with '_score' (excluding the overall 'demand_score')
        feature_cols = [col for col in df.columns if col.endswith('_score') and col != 'demand_score']
        # Fill missing values and standardize the features
        data = df[feature_cols].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        # Run PCA to extract the first principal component
        pca = PCA(n_components=1)
        pca.fit(scaled_data)
        loadings = pca.components_[0]
        # Map each feature to its loading
        weights = dict(zip(feature_cols, loadings))
        # Flip sign for features that should have a negative impact
        for feat in ['radius_score', 'cs_total_score']:
            if feat in weights and weights[feat] > 0:
                weights[feat] *= -1
        return weights

# Dummy implementations for the UI components (replace with your actual modules if available)
def render_weight_analysis_ui(df, optimizer):
    st.sidebar.markdown("### Weight Analysis UI")
    st.sidebar.info("This panel would allow interactive weight adjustments and visualizations.")

# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="EV Charging Station Demand Map",
    layout="wide"
)

st.title("EV Charging Station Demand Score Map with ML Analysis & Weight Optimization")
st.markdown(
    "This app identifies optimal locations for new EV charging stations by analyzing multiple factors. "
    "It computes a **demand score** for each candidate location using features such as distance to the nearest charger, "
    "nearby charger density, traffic, population, income, and zoning suitability.\n\n"
    "Users can choose between using automatically optimized weights (via PCA) or manually adjusting the parameters. "
    "Note: PCA’s loadings are arbitrarily signed; in our PCA mode we flip the sign for features (e.g. `radius_score` and `cs_total_score`) "
    "that should have a negative impact on demand."
)

# ---------------------------------------------------------------------
# Helper Functions and Data Loading
# ---------------------------------------------------------------------
def map_zoning_category(zone_code):
    """Map zone codes to categories."""
    zoning_categories = {
        'Commercial': ['CC', 'CN', 'CV', 'CP', 'CR', 'CCPD'],
        'Office': ['CO'],
        'Residential High': ['RH', 'RM-3', 'RM-4'],
        'Residential Medium': ['RM-2', 'RM-1'],
        'Residential Low': ['RS', 'RL'],
        'Residential Mixed': ['RMX'],
        'Industrial': ['IP', 'IL', 'IH', 'IS', 'IBT'],
        'Mixed Use': ['MU', 'EMX'],
        'Agricultural': ['AG', 'AR'],
        'Open Space': ['OS'],
        'Planned': ['BLPD', 'MBPD', 'GQPD', 'MPD', 'CUPD', 'LJPD', 'LJSPD'],
        'Transit': ['OTOP', 'OTRM', 'OTCC'],
        'Other': ['UNZONED'],
    }
    if isinstance(zone_code, str):
        for category, prefixes in zoning_categories.items():
            if any(zone_code.startswith(prefix) for prefix in prefixes):
                return category
    return 'Other'

def convert_timestamps(df):
    """将DataFrame中的时间戳列转换为字符串以便JSON序列化"""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_data():
    """
    Load and prepare the GeoDataFrame with merged features.
    Replace with your actual data file(s).
    """
    merged_features = pd.read_csv("features.csv")
    merged_features['geometry'] = gpd.GeoSeries.from_wkt(merged_features['geometry'])
    merged_features = gpd.GeoDataFrame(merged_features, geometry='geometry')
    merged_features.set_crs(epsg=4326, inplace=True)
    merged_features = convert_timestamps(merged_features)

    # Calculate zoning suitability score
    zoning_suitability = {
        'Commercial': 1.0, 
        'Office': 0.8, 
        'Mixed Use': 0.7, 
        'Transit': 0.6, 
        'Industrial': 0.5, 
        'Planned': 0.5, 
        'Other': 0.4, 
        'Residential High': 0.3, 
        'Residential Medium': 0.2, 
        'Residential Low': 0.1, 
        'Agricultural': 0.0, 
        'Open Space': 0.0, 
        'Multiple': 0.7
    }
    merged_features['zoning_score'] = merged_features['zone_type'].map(zoning_suitability) * 0.1
    
    # Compute feature scores using default (expert-assigned) weights
    merged_features['distance_score'] = merged_features['percentile_distance_to_nearest_charger'] * 0.5
    merged_features['radius_score'] = merged_features['percentile_chargers_in_radius'] * -0.3
    merged_features['cs_total_score'] = merged_features['percentile_cs_total'] * -0.2
    merged_features['traffic_score'] = merged_features['percentile_traffic'] * 0.1
    merged_features['population_score'] = merged_features['percentile_Population'] * 0.2
    merged_features['income_score'] = merged_features['percentile_Median Income'] * 0.1
    
    # Initial overall demand score (using default weights)
    merged_features['demand_score'] = (
        merged_features['distance_score'] +
        merged_features['radius_score'] +
        merged_features['cs_total_score'] +
        merged_features['traffic_score'] +
        merged_features['population_score'] +
        merged_features['income_score'] +
        merged_features['zoning_score']
    )
    
    return merged_features

@st.cache_data
def load_supporting_data():
    """Load zoning, parking, and lots data."""
    zoning_data = gpd.read_file("shapefiles/zoning_datasd.geojson")
    zoning_data = zoning_data.to_crs(epsg=4326)
    if 'zone_type' not in zoning_data.columns and 'zone_name' in zoning_data.columns:
        zoning_data['zone_type'] = zoning_data['zone_name'].apply(map_zoning_category)
    public_parking_gdf = gpd.read_file("parking_data.geojson")
    public_parking_gdf = public_parking_gdf.to_crs(epsg=4326)
    lots = gpd.read_file("lots.geojson")
    lots = lots.to_crs(epsg=4326)
    zoning_data = convert_timestamps(zoning_data)
    public_parking_gdf = convert_timestamps(public_parking_gdf)
    lots = convert_timestamps(lots)
    return zoning_data, public_parking_gdf, lots

@st.cache_data
def analyze_scores_with_ml(_data):
    """
    Use ML techniques (clustering, quantile thresholds) to analyze scores.
    """
    data = _data.copy()
    score_columns = [col for col in data.columns if col.endswith('_score') and col != 'demand_score']
    if not score_columns:
        return {
            'demand_thresholds': [0.33, 0.66],
            'factor_thresholds': {},
            'clusters': None,
            'cluster_descriptions': {}
        }
    features = data[score_columns].fillna(0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    n_clusters = min(5, len(data) // 10) if len(data) > 10 else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    data['cluster'] = clusters
    cluster_profiles = data.groupby('cluster')[score_columns + ['demand_score']].mean()
    cluster_profiles = cluster_profiles.sort_values('demand_score', ascending=False)
    cluster_descriptions = {}
    for i, (cluster_idx, profile) in enumerate(cluster_profiles.iterrows()):
        if i == 0:
            desc = "High Priority"
        elif i == 1 and n_clusters > 2:
            desc = "Medium Priority"
        else:
            desc = "Low Priority"
        factor_scores = profile[score_columns].sort_values(ascending=False)
        top_factors = factor_scores.index[:3].tolist()
        factor_desc = [f.replace('_score', '').replace('_', ' ').title() for f in top_factors]
        cluster_descriptions[cluster_idx] = {
            'description': desc,
            'top_factors': factor_desc,
            'avg_demand_score': profile['demand_score']
        }
    demand_scores = data['demand_score'].dropna()
    if len(demand_scores) > 0:
        demand_thresholds = [
            demand_scores.quantile(0.33),
            demand_scores.quantile(0.66)
        ]
    else:
        demand_thresholds = [0.33, 0.66]
    factor_thresholds = {}
    for factor in score_columns:
        values = data[factor].dropna()
        if len(values) > 0:
            factor_thresholds[factor] = [
                values.quantile(0.33),
                values.quantile(0.66)
            ]
    return {
        'demand_thresholds': demand_thresholds,
        'factor_thresholds': factor_thresholds,
        'clusters': clusters.tolist(),
        'cluster_descriptions': cluster_descriptions
    }

def generate_ml_explanations(recommendations_df, ml_analysis):
    """
    Generate HTML explanation for recommendations.
    """
    demand_thresholds = ml_analysis['demand_thresholds']
    cluster_descriptions = ml_analysis.get('cluster_descriptions', {})
    rec_counts = {
        "Highly Recommended": 0,
        "Recommended": 0,
        "Not Recommended": 0
    }
    recommendations_with_status = recommendations_df.copy()
    recommendations_with_status['recommendation_status'] = recommendations_with_status.apply(
        lambda row: get_recommendation_status(
            row['demand_score'], 
            demand_thresholds,
            cluster_descriptions.get(row.get('cluster'))
        ),
        axis=1
    )
    status_counts = recommendations_with_status['recommendation_status'].value_counts()
    for status, count in status_counts.items():
        if status in rec_counts:
            rec_counts[status] = count
    top_rec = recommendations_with_status.iloc[0]
    score_columns = [col for col in recommendations_df.columns if col.endswith('_score') and col != 'demand_score']
    explanation_html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h4 style="margin-top: 0; color: #333;">Analysis of Recommendations</h4>
        <p><strong>Overview:</strong> Our machine learning analysis has evaluated {len(recommendations_df)} locations 
        for EV charging station placement potential. Of these, <span style="color: green; font-weight: bold;">
        {rec_counts['Highly Recommended']} locations are highly recommended</span>, 
        {rec_counts['Recommended']} are recommended, and {rec_counts['Not Recommended']} are not recommended 
        based on our predictive model.</p>
        <p><strong>Top Location (ID: {top_rec.osmid}):</strong> This {top_rec.zone_type} location in ZIP code {top_rec.zip} 
        has a demand score of {top_rec.demand_score:.2f} and is <span style="font-weight: bold;">
        {top_rec.recommendation_status}</span>.</p>
        <p><strong>Key Location Insights:</strong></p>
        <ul style="margin-top: 0.5em;">
    """
    for i, (idx, row) in enumerate(recommendations_with_status.head(3).iterrows()):
        explanation_html += f"""
            <li style="margin-bottom: 1em;"><strong>Rank #{i+1} (ID: {row.osmid}):</strong> {row.recommendation_status} - 
            Demand score: {row.demand_score:.2f}
        """
        zone_descriptions = {
            'Commercial': 'commercial district ideal for public charging',
            'Office': 'office area with good daytime demand',
            'Mixed Use': 'mixed-use area with diverse charging needs',
            'Transit': 'transit hub with consistent visitor flow',
            'Industrial': 'industrial zone with specialized charging needs',
            'Planned': 'planned development area',
            'Residential High': 'high-density residential neighborhood',
            'Residential Medium': 'medium-density residential area',
            'Residential Low': 'low-density residential community',
            'Agricultural': 'agricultural zone with limited charging opportunities',
            'Open Space': 'open space with minimal infrastructure',
            'Multiple': 'diverse zoning area with mixed usage',
            'Other': 'miscellaneous zoning designation'
        }
        zone_description = zone_descriptions.get(row.zone_type, 'area')
        explanation_html += f"""
            <br>Located in a <strong>{zone_description}</strong>
        """
        if 'cluster' in row and row['cluster'] in cluster_descriptions:
            cluster_info = cluster_descriptions[row['cluster']]
            explanation_html += f"""
                <br>Part of {cluster_info['description']} cluster based on {', '.join(cluster_info['top_factors'][:2])}
            """
        explanation_html += "<ul style='margin-top: 0.5em;'>"
        if 'distance_to_nearest_charger' in row and pd.notnull(row['distance_to_nearest_charger']):
            distance = row['distance_to_nearest_charger']
            if distance < 200:
                distance_desc = f"Very close to existing infrastructure ({distance:.0f}m away)"
                color = "red"
            elif distance < 500:
                distance_desc = f"Moderately distant from existing chargers ({distance:.0f}m away)"
                color = "orange"
            else:
                distance_desc = f"Far from existing infrastructure ({distance:.0f}m away)"
                color = "green"
            explanation_html += f"""
                <li><span style="color: {color};">{distance_desc}</span> 
                (score: {row.get('distance_score', 0):.2f})</li>
            """
        if 'chargers_in_radius' in row and pd.notnull(row['chargers_in_radius']):
            chargers = row['chargers_in_radius']
            if chargers == 0:
                radius_desc = "No nearby charging stations in the immediate area"
                color = "green"
            elif chargers < 3:
                radius_desc = f"Few nearby charging stations ({chargers} within radius)"
                color = "green"
            else:
                radius_desc = f"Multiple nearby charging stations ({chargers} within radius)"
                color = "orange"
            explanation_html += f"""
                <li><span style="color: {color};">{radius_desc}</span> 
                (score: {row.get('radius_score', 0):.2f})</li>
            """
        if 'traffic' in row and pd.notnull(row['traffic']) and row['traffic'] > 0:
            traffic = row['traffic']
            if traffic < 5000:
                traffic_desc = f"Low traffic volume area ({traffic:.0f} vehicles)"
                color = "orange"
            elif traffic < 15000:
                traffic_desc = f"Moderate traffic flow ({traffic:.0f} vehicles)"
                color = "green"
            else:
                traffic_desc = f"High traffic volume location ({traffic:.0f} vehicles)"
                color = "green"
            explanation_html += f"""
                <li><span style="color: {color};">{traffic_desc}</span> 
                (score: {row.get('traffic_score', 0):.2f})</li>
            """
        if 'Population' in row and pd.notnull(row['Population']):
            population = row['Population']
            if population < 2000:
                pop_desc = f"Low population density area ({population:.0f} people)"
                color = "orange"
            elif population < 5000:
                pop_desc = f"Medium population density ({population:.0f} people)"
                color = "green"
            else:
                pop_desc = f"High population density location ({population:.0f} people)"
                color = "green"
            explanation_html += f"""
                <li><span style="color: {color};">{pop_desc}</span> 
                (score: {row.get('population_score', 0):.2f})</li>
            """
        if 'Median Income' in row and pd.notnull(row['Median Income']):
            income = row['Median Income']
            if income < 50000:
                income_desc = f"Lower income area (${income:.0f} median income)"
                color = "orange"
            elif income < 100000:
                income_desc = f"Middle income neighborhood (${income:.0f} median income)"
                color = "green"
            else:
                income_desc = f"Higher income location (${income:.0f} median income)"
                color = "green"
            explanation_html += f"""
                <li><span style="color: {color};">{income_desc}</span> 
                (score: {row.get('income_score', 0):.2f})</li>
            """
        if row['recommendation_status'] == "Highly Recommended":
            recommendation = "This location should be considered a top priority for new charging infrastructure."
        elif row['recommendation_status'] == "Recommended":
            recommendation = "This location is suitable for charging infrastructure in medium-term planning."
        else:
            recommendation = "This location is not currently recommended for charging infrastructure."
        explanation_html += f"""
                <li><strong>Recommendation:</strong> <span style="color: #333;">{recommendation}</span></li>
            </ul>
        </li>
        """
    explanation_html += """</ul>
        
        <p><strong>Machine Learning Insights:</strong></p>
        <ul style="margin-top: 0.5em;">
    """
    if cluster_descriptions:
        n_clusters = len(cluster_descriptions)
        explanation_html += f"""
            <li>Our machine learning algorithm identified {n_clusters} distinct location profiles based on multiple factors.</li>
        """
        top_cluster_idx = next(iter(cluster_descriptions))
        top_cluster = cluster_descriptions[top_cluster_idx]
        explanation_html += f"""
            <li>The highest priority cluster shows strong potential with average demand score of 
            {top_cluster['avg_demand_score']:.2f}, characterized by {', '.join(top_cluster['top_factors'][:2])}.</li>
        """
    explanation_html += f"""
        <li>Locations with demand scores above {demand_thresholds[1]:.2f} are considered high priority 
        and below {demand_thresholds[0]:.2f} are low priority based on our quantile analysis.</li>
    """
    if score_columns:
        avg_scores = recommendations_df[score_columns].mean().sort_values(ascending=False)
        top_factors = avg_scores.index[:3]
        factor_descs = [col.replace('_score', '').replace('_', ' ').title() for col in top_factors]
        explanation_html += f"""
            <li>The most influential factors across all recommendations are {', '.join(factor_descs)}.</li>
        """
    explanation_html += """
        <li>Distance from existing charging infrastructure is the strongest predictor of need, with a weight of 0.5 in our model.</li>
        <li>Population density (weight: 0.2) and nearby charger saturation (weight: -0.3) are also key determinants.</li>
    """
    explanation_html += """</ul>
        
        <p><strong>Recommendation Summary:</strong> Based on our machine learning analysis, we recommend prioritizing 
        the highly recommended locations for immediate consideration, followed by the recommended locations as secondary options. 
        Locations marked as not recommended should be excluded from immediate planning.</p>
        
        <p><em>Methodology: Our model uses a weighted sum of factors including distance to the nearest charger (0.5), 
        nearby chargers (-0.3), existing charging points (-0.2), traffic volume (0.1), population density (0.2), 
        income level (0.1), and zoning suitability (0.1), combined with K-means clustering to identify patterns.</em></p>
    </div>
    """
    return explanation_html, recommendations_with_status

def get_recommendation_status(demand_score, thresholds, cluster_desc=None):
    if demand_score < thresholds[0]:
        status = "Not Recommended"
    elif demand_score > thresholds[1]:
        status = "Highly Recommended"
    else:
        status = "Recommended"
    if cluster_desc:
        if cluster_desc['description'] == "High Priority":
            status = "Highly Recommended"
        elif cluster_desc['description'] == "Low Priority":
            if status == "Highly Recommended":
                status = "Recommended"
    return status

def create_combined_map(zoning_data, public_parking_gdf, lots, merged_features):
    combined_map = folium.Map(location=[32.7157, -117.1611], zoom_start=12, tiles="cartodbpositron")
    color_mapping = {
        'Commercial': '#4A90E2',
        'Office': '#50C878',
        'Residential High': '#E74C3C',
        'Residential Medium': '#F4D03F',
        'Residential Low': '#F9E79F',
        'Residential Mixed': '#FF69B4',
        'Industrial': '#95A5A6',
        'Mixed Use': '#2CCBCB',
        'Agricultural': '#DAA520',
        'Open Space': '#87CEFA',
        'Planned': '#00CED1',
        'Transit': '#9370DB',
        'Other': '#5D6D7E',
        'Multiple': '#9932CC'
    }
    def zoning_style(feature):
        properties = feature['properties']
        if 'zone_type' in properties:
            zone_type = properties['zone_type']
        elif 'zone_name' in properties:
            zone_name = properties['zone_name']
            zone_type = map_zoning_category(zone_name)
        else:
            zone_type = 'Other'
        color = color_mapping.get(zone_type, 'gray')
        return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}
    def get_coords(geometry):
        if isinstance(geometry, (Point, Polygon, MultiPolygon)):
            centroid = geometry.centroid
            return centroid.y, centroid.x
        return np.nan, np.nan
    zoning_layer = folium.FeatureGroup(name='Zoning Areas')
    parking_layer = folium.FeatureGroup(name='Public Parking Lots')
    zoned_parking_layer = folium.FeatureGroup(name='Parking by Zone Type')
    charger_distance_layer = folium.FeatureGroup(name='Distance to Nearest Charger')
    st.sidebar.write("Data Loading Status:")
    if len(zoning_data) > 0:
        st.sidebar.success(f"✅ Zoning data loaded: {len(zoning_data)} features")
        first_feature = zoning_data.iloc[0]
        debug_info = {col: str(first_feature[col])[:50] for col in zoning_data.columns}
        if st.sidebar.checkbox("Show zoning data details", False):
            st.sidebar.write("First zoning feature properties:", debug_info)
    else:
        st.sidebar.error("❌ Zoning data is empty")
    try:
        zoning_geojson = zoning_data.__geo_interface__
        folium.GeoJson(
            zoning_geojson,
            style_function=zoning_style,
            name='Zoning Areas'
        ).add_to(zoning_layer)
    except Exception as e:
        st.sidebar.error(f"Error adding zoning layer: {e}")
    try:
        for _, row in public_parking_gdf.iterrows():
            if row.geometry.geom_type == "Point":
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color="#004080",
                    fill=True,
                    fill_color="#0066CC",
                    fill_opacity=0.8,
                    weight=2,
                    tooltip="Public Parking Lot",
                    popup=folium.Popup(f"Public Parking Lot<br>Type: {row.get('parking', 'Unknown')}", max_width=250),
                ).add_to(parking_layer)
            elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x: {
                        "color": "#004080",
                        "fillColor": "#0066CC",
                        "weight": 2,
                        "fillOpacity": 0.6,
                    },
                    tooltip="Non-Private Parking Lot"
                ).add_to(parking_layer)
    except Exception as e:
        st.sidebar.error(f"Error adding parking layer: {e}")
    try:
        for _, row in lots.iterrows():
            zone_type = row['zone_type'] if 'zone_type' in row else "Unknown"
            zone_color = color_mapping.get(zone_type, 'gray')
            if row.geometry.geom_type == "Point":
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color=zone_color,
                    fill=True,
                    fill_color=zone_color,
                    fill_opacity=0.9,
                    weight=2,
                    tooltip=f"Zone: {zone_type}",
                    popup=folium.Popup(f"Zone Type: {zone_type}<br>OSMID: {row.get('osmid', 'N/A')}", max_width=250),
                ).add_to(zoned_parking_layer)
            elif row.geometry.geom_type in ["Polygon", "MultiPolygon"]:
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, color=zone_color: {"color": color, "fillColor": color, "weight": 1, "fillOpacity": 0.4},
                    tooltip=f"Zone: {zone_type}"
                ).add_to(zoned_parking_layer)
    except Exception as e:
        st.sidebar.error(f"Error adding zoned parking layer: {e}")
    try:
        for _, row in merged_features.iterrows():
            lat, lon = get_coords(row.geometry)
            if np.isnan(lat) or np.isnan(lon):
                continue
            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.7,
                weight=2,
                tooltip=f"OSMID: {row.osmid} (Click for details)",
                popup=folium.Popup(f"<b>Charging Station Data</b><br>OSMID: {row.osmid}<br>Zone Name: {row.zone_name}<br>Distance to Charger: {row.distance_to_nearest_charger:.1f}m", max_width=300),
            ).add_to(charger_distance_layer)
            folium.Circle(
                location=(lat, lon),
                radius=row.distance_to_nearest_charger,
                color="blue",
                weight=2,
                opacity=0.3,
                fill=True,
                fill_opacity=0.01,
            ).add_to(charger_distance_layer)
    except Exception as e:
        st.sidebar.error(f"Error adding charger distance layer: {e}")
    opacity = 0.8
    legend_html = f"""
    <div style="position: fixed; 
                 top: 10px; left: 10px; 
                 width: 200px; height: auto; 
                 background-color: rgba(255, 255, 255, {opacity}); 
                 border:2px solid grey; 
                 z-index: 9999; 
                 font-size:14px;
                 padding: 10px;">
        <b>Zoning Categories</b><br>
        <div style="line-height: 1.5;">
    """
    for category, color in color_mapping.items():
        legend_html += f'<div><i style="background:{color}; width: 20px; height: 20px; display: inline-block;"></i> {category}</div>'
    legend_html += "</div></div>"
    combined_map.get_root().html.add_child(folium.Element(legend_html))
    charger_legend_html = f"""
    <div style="position: fixed; 
                 bottom: 10px; right: 10px; 
                 width: 220px; height: auto; 
                 background-color: rgba(255, 255, 255, {opacity}); 
                 border:2px solid grey; 
                 z-index: 9999; 
                 font-size:14px;
                 padding: 10px;">
        <b>Charger Distance Layer</b><br>
        <div style="line-height: 1.5;">
        <div><i style="background:red; width: 10px; height: 10px; border-radius: 50%; display: inline-block;"></i> Parking Spot</div>
        <div><i style="border: 2px solid blue; width: 16px; height: 16px; border-radius: 50%; display: inline-block;"></i> Distance to Nearest Charger</div>
        </div>
    </div>
    """
    combined_map.get_root().html.add_child(folium.Element(charger_legend_html))
    zoning_layer.add_to(combined_map)
    parking_layer.add_to(combined_map)
    zoned_parking_layer.add_to(combined_map)
    charger_distance_layer.add_to(combined_map)
    MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(combined_map)
    folium.LayerControl(position='topright', collapsed=True).add_to(combined_map)
    custom_layer_control = '''
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        setTimeout(function() {
            var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            if (checkboxes.length >= 4) {
                checkboxes[1].click();
                checkboxes[2].click();
                checkboxes.forEach(function(checkbox, index) {
                    checkbox.addEventListener('change', function() {
                        console.log("Layer " + index + " is now: " + (this.checked ? "on" : "off"));
                        var enabledLayers = Array.from(checkboxes).filter(cb => cb.checked).length;
                        if (enabledLayers > 2) {
                            var tipElement = document.getElementById('map-layer-tip');
                            if (!tipElement) {
                                tipElement = document.createElement('div');
                                tipElement.id = 'map-layer-tip';
                                tipElement.style.position = 'fixed';
                                tipElement.style.bottom = '40px';
                                tipElement.style.left = '50%';
                                tipElement.style.transform = 'translateX(-50%)';
                                tipElement.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                                tipElement.style.color = 'white';
                                tipElement.style.padding = '8px 12px';
                                tipElement.style.borderRadius = '4px';
                                tipElement.style.fontSize = '14px';
                                tipElement.style.zIndex = '1000';
                                document.body.appendChild(tipElement);
                            }
                            tipElement.textContent = 'Tip: Opening multiple layers at the same time may cause click conflicts. It is recommended to view layers one at a time.';
                            tipElement.style.display = 'block';
                            setTimeout(function() {
                                tipElement.style.display = 'none';
                            }, 3000);
                        }
                    });
                });
            }
        }, 1000);
    });
    </script>
    '''
    combined_map.get_root().html.add_child(folium.Element(custom_layer_control))
    title_html = '''
    <h3 align="center" style="font-size:20px; font-family: 'Arial'; margin-top: 10px;">
        <b>EV Charging Station Geographic Context</b>
    </h3>
    '''
    combined_map.get_root().html.add_child(folium.Element(title_html))
    return combined_map

def create_radar_chart(top_recommendations, score_columns):
    if len(top_recommendations) == 0:
        return None, "No recommendations data available"
    top_loc = top_recommendations.iloc[0]
    available_score_cols = [col for col in score_columns if col in top_loc.index]
    if len(available_score_cols) < 3:
        return None, "Not enough score dimensions for radar chart (need at least 3)"
    radar_data = top_loc[available_score_cols].copy()
    for col in available_score_cols:
        if not pd.api.types.is_numeric_dtype(radar_data[col]):
            try:
                radar_data[col] = pd.to_numeric(radar_data[col])
            except:
                return None, f"Column {col} contains non-numeric data"
    max_abs = max(abs(radar_data.min()), abs(radar_data.max()))
    if max_abs > 0:
        radar_data = radar_data / max_abs
    column_labels = {
        'distance_score': 'Distance (0.5)', 
        'radius_score': 'Nearby Chargers (-0.3)', 
        'cs_total_score': 'Charging Points (-0.2)', 
        'traffic_score': 'Traffic (0.1)', 
        'population_score': 'Population (0.2)', 
        'income_score': 'Income (0.1)',
        'zoning_score': 'Zoning (0.1)'
    }
    categories = [column_labels.get(col, col.replace('_score', '').replace('_', ' ').title()) for col in available_score_cols]
    values = radar_data.tolist()
    values.append(values[0])
    categories.append(categories[0])
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    if len(angles) != len(values):
        return None, f"Dimension mismatch: angles ({len(angles)}) and values ({len(values)}) have different lengths"
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    plt.xticks(angles[:-1], categories[:-1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
    ax.grid(True)
    plt.title(f'Factor Profile for Top Location (ID: {top_loc.osmid})', pad=20)
    return fig, None

def plot_radar_in_streamlit(top_recommendations, score_columns):
    try:
        fig, error = create_radar_chart(top_recommendations, score_columns)
        if error:
            st.warning(f"Could not create radar chart: {error}")
            return
        if fig:
            st.pyplot(fig)
            top_loc = top_recommendations.iloc[0]
            st.markdown(f"**ID**: {top_loc.osmid} | **ZIP**: {top_loc.zip} | **Zone**: {top_loc.zone_type}")
            scores = {col: top_loc[col] for col in score_columns if col in top_loc}
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if sorted_scores:
                strongest = sorted_scores[0]
                strongest_factor = strongest[0].replace('_score', '').replace('_', ' ').title()
                factor_descriptions = {
                    'distance_score': f"Far from existing chargers ({top_loc.get('distance_to_nearest_charger', 'N/A')}m)",
                    'radius_score': f"Few nearby charging stations ({top_loc.get('chargers_in_radius', 'N/A')} in radius)",
                    'cs_total_score': f"Limited existing charging infrastructure ({top_loc.get('cs_total', 'N/A')} points)",
                    'traffic_score': f"High traffic volume ({top_loc.get('traffic', 'N/A')})",
                    'population_score': f"Dense population area ({top_loc.get('Population', 'N/A')})",
                    'income_score': f"Higher income neighborhood (${top_loc.get('Median Income', 'N/A')})",
                    'zoning_score': f"Suitable zoning type for charging stations"
                }
                strongest_desc = factor_descriptions.get(strongest[0], strongest_factor)
                st.markdown(f"**Strongest Factor**: {strongest_factor} ({strongest[1]:.2f}) - {strongest_desc}")
                if len(sorted_scores) > 1:
                    weakest = sorted_scores[-1]
                    weakest_factor = weakest[0].replace('_score', '').replace('_', ' ').title()
                    weakest_desc = factor_descriptions.get(weakest[0], weakest_factor)
                    st.markdown(f"**Lowest Factor**: {weakest_factor} ({weakest[1]:.2f}) - {weakest_desc}")
        else:
            st.info("Not enough data to create radar chart")
    except Exception as e:
        st.warning(f"Error creating radar chart: {e}")

# ---------------------------------------------------------------------
# Main Application Logic
# ---------------------------------------------------------------------
try:
    merged_features = load_data()
    try:
        zoning_data, public_parking_gdf, lots = load_supporting_data()
        available_zips = sorted(merged_features['zip'].unique())
        
        # Create main tabs for Overview Map and Recommendations
        tab1, tab2 = st.tabs(["Overview Map", "Recommendations"])
        
        with tab1:
            st.header("EV Charging Station Geographic Overview")
            st.markdown("""
            This map provides a comprehensive view of the area's charging infrastructure landscape:
            - **Zoning Areas**: Color-coded by zone type (commercial, residential, etc.)
            - **Public Parking Lots**: All available public parking
            - **Parking by Zone Type**: Parking lots colored by their zone classification
            - **Distance to Nearest Charger**: Showing existing charger proximity
            
            Use the layer control (top right of map) to toggle different data layers.
            """)
            combined_map = create_combined_map(zoning_data, public_parking_gdf, lots, merged_features)
            folium_static(combined_map, width=1000, height=600)
            st.markdown("""
            ### Map Legend and Instructions
            - Toggle layers using the control in the top-right corner
            - Click on any point for detailed information
            - The colored areas represent different zoning types
            - Red circles show parking locations
            - Blue circles show distance to the nearest existing charger
            """)
        
        with tab2:
            st.header("EV Charging Station Recommendations")
            st.sidebar.header("Filter Options")
            zip_options = ["All ZIP Codes"] + [str(z) for z in available_zips]
            selected_zip = st.sidebar.selectbox("Select ZIP Code:", zip_options)
            zip_filter = None if selected_zip == "All ZIP Codes" else int(selected_zip)
            num_recommendations = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=10, value=2)
            
            # Define map dimensions
            map_width = 800
            map_height = 500
            
            # Weight Mode: PCA Optimized vs. Manual Adjustment
            st.sidebar.header("Weight Mode")
            weight_mode = st.sidebar.radio(
                "Choose weight mode:",
                options=["PCA Optimized", "Manual Adjustment"],
                index=0
            )
            
            if weight_mode == "Manual Adjustment":
                manual_weights = {}
                manual_weights['distance_score'] = st.sidebar.slider("Distance Weight (0.5)", 0.0, 1.0, 0.5, 0.05)
                manual_weights['radius_score'] = st.sidebar.slider("Radius Weight (-0.3)", -1.0, 0.0, -0.3, 0.05)
                manual_weights['cs_total_score'] = st.sidebar.slider("Charging Points Weight (-0.2)", -1.0, 0.0, -0.2, 0.05)
                manual_weights['traffic_score'] = st.sidebar.slider("Traffic Weight (0.1)", 0.0, 1.0, 0.1, 0.05)
                manual_weights['population_score'] = st.sidebar.slider("Population Weight (0.2)", 0.0, 1.0, 0.2, 0.05)
                manual_weights['income_score'] = st.sidebar.slider("Income Weight (0.1)", 0.0, 1.0, 0.1, 0.05)
                manual_weights['zoning_score'] = st.sidebar.slider("Zoning Weight (0.1)", 0.0, 1.0, 0.1, 0.05)
                chosen_weights = manual_weights
            else:
                pca_weights = WeightOptimizer.optimize_weights(merged_features)
                chosen_weights = pca_weights
                st.session_state.optimized_weights = {"Consensus": pca_weights}
            
            if st.sidebar.button("Show Recommendations"):
                with st.spinner("Preparing map and running ML analysis..."):
                    if zip_filter:
                        filtered_data = merged_features[merged_features['zip'] == zip_filter]
                    else:
                        filtered_data = merged_features
                    
                    # Recalculate demand score using the chosen weights
                    merged_features['new_demand_score'] = 0.0
                    for feat, w in chosen_weights.items():
                        if feat in merged_features.columns:
                            merged_features['new_demand_score'] += merged_features[feat].fillna(0) * w
                    
                    # Sort recommendations using new_demand_score
                    top_recommendations = filtered_data.sort_values('new_demand_score', ascending=False).head(num_recommendations)
                    
                    if st.sidebar.checkbox("Use ML Analysis", value=True):
                        ml_analysis = analyze_scores_with_ml(filtered_data)
                        ml_explanation, top_recommendations = generate_ml_explanations(top_recommendations, ml_analysis)
                    else:
                        ml_analysis = None
                        ml_explanation = None
                    
                    if top_recommendations.shape[0] > 0:
                        if all(hasattr(geom, 'y') and hasattr(geom, 'x') for geom in top_recommendations.geometry):
                            min_lat = top_recommendations.geometry.y.min() - 0.05
                            max_lat = top_recommendations.geometry.y.max() + 0.05
                            min_lon = top_recommendations.geometry.x.min() - 0.05
                            max_lon = top_recommendations.geometry.x.max() + 0.05
                            center_lat = (min_lat + max_lat) / 2
                            center_lon = (min_lon + max_lon) / 2
                            map_center = [center_lat, center_lon]
                            sw = [min_lat, min_lon]
                            ne = [max_lat, max_lon]
                        else:
                            centroids = top_recommendations.geometry.centroid
                            min_lat = centroids.y.min() - 0.05
                            max_lat = centroids.y.max() + 0.05
                            min_lon = centroids.x.min() - 0.05
                            max_lon = centroids.x.max() + 0.05
                            center_lat = (min_lat + max_lat) / 2
                            center_lon = (min_lon + max_lon) / 2
                            map_center = [center_lat, center_lon]
                            sw = [min_lat, min_lon]
                            ne = [max_lat, max_lon]
                    else:
                        map_center = [32.7157, -117.1611]
                        sw, ne = None, None
                    
                    california_bounds = [
                        [32.5, -124.5],
                        [42.0, -114.0]
                    ]
                    m = folium.Map(
                        location=map_center,
                        zoom_start=12,
                        tiles='CartoDB positron',
                        min_zoom=7,
                        max_zoom=18,
                        zoom_control=True,
                        scrollWheelZoom=True,
                        dragging=True
                    )
                    folium.Rectangle(
                        bounds=california_bounds,
                        color='gray',
                        weight=1,
                        fill=False,
                        opacity=0.3,
                        tooltip="California state bounds (approximate)"
                    ).add_to(m)
                    if sw and ne:
                        m.fit_bounds([sw, ne])
                    
                    script = f"""
                    <script>
                        var map = document.querySelector('#{m.get_name()}').map;
                        var southWest = L.latLng(32.0, -125.0);
                        var northEast = L.latLng(42.5, -113.0);
                        var californiaBounds = L.latLngBounds(southWest, northEast);
                        map.setMaxBounds(californiaBounds);
                        map.on('drag', function() {{
                            map.panInsideBounds(californiaBounds, {{ animate: false }});
                        }});
                    </script>
                    """
                    m.get_root().html.add_child(folium.Element(script))
                    colormap = cm.linear.YlOrRd_09.scale(filtered_data['new_demand_score'].min(), filtered_data['new_demand_score'].max())
                    colormap.caption = "Demand Score"
                    colormap.add_to(m)
                    
                    all_parking_fg = folium.FeatureGroup(name="All Parking Lots", show=False)
                    if sw and ne:
                        visible_data = filtered_data[
                            (filtered_data.geometry.centroid.y >= min_lat - 0.1) & 
                            (filtered_data.geometry.centroid.y <= max_lat + 0.1) & 
                            (filtered_data.geometry.centroid.x >= min_lon - 0.1) & 
                            (filtered_data.geometry.centroid.x <= max_lon + 0.1)
                        ]
                    else:
                        visible_data = filtered_data
                    for _, row in visible_data.iterrows():
                        color = colormap(row['new_demand_score'])
                        tooltip = f"ID: {row.osmid}, Score: {row.new_demand_score:.2f}, ZIP: {row.zip}"
                        if hasattr(row.geometry, 'y') and hasattr(row.geometry, 'x'):
                            folium.Circle(
                                location=(row.geometry.y, row.geometry.x),
                                radius=15,
                                color=color,
                                fill=True,
                                fill_opacity=0.6,
                                tooltip=tooltip
                            ).add_to(all_parking_fg)
                        else:
                            try:
                                folium.GeoJson(
                                    row.geometry,
                                    style_function=lambda x, color=color: {
                                        'fillColor': color,
                                        'color': color,
                                        'weight': 1,
                                        'fillOpacity': 0.6
                                    },
                                    tooltip=tooltip
                                ).add_to(all_parking_fg)
                            except Exception as e:
                                pass
                    recommendations_fg = folium.FeatureGroup(name="Recommended Parking Lots", show=True)
                    for i, (_, row) in enumerate(top_recommendations.iterrows()):
                        if 'recommendation_status' in row:
                            if row['recommendation_status'] == "Highly Recommended":
                                highlight_color = '#FF0000'
                            elif row['recommendation_status'] == "Recommended":
                                highlight_color = '#FF8C00'
                            else:
                                highlight_color = '#4682B4'
                        else:
                            highlight_color = '#FF4500'
                        tooltip_content = [
                            f"Rank #{i+1}",
                            f"ID: {row.osmid}",
                            f"Score: {row.new_demand_score:.2f}",
                            f"ZIP: {row.zip}",
                            f"Zone Type: {row.zone_type if 'zone_type' in row else 'N/A'}"
                        ]
                        if 'recommendation_status' in row:
                            tooltip_content.insert(1, f"Status: {row.recommendation_status}")
                        tooltip = "<br>".join(tooltip_content)
                        if hasattr(row.geometry, 'y') and hasattr(row.geometry, 'x'):
                            folium.Circle(
                                location=(row.geometry.y, row.geometry.x),
                                radius=25,
                                color=highlight_color,
                                weight=3,
                                fill=True,
                                fill_opacity=0.7,
                                tooltip=tooltip
                            ).add_to(recommendations_fg)
                        else:
                            try:
                                folium.GeoJson(
                                    row.geometry,
                                    style_function=lambda x, color=highlight_color: {
                                        'fillColor': color,
                                        'color': color,
                                        'weight': 3,
                                        'fillOpacity': 0.7
                                    },
                                    tooltip=tooltip
                                ).add_to(recommendations_fg)
                            except Exception as e:
                                pass
                    all_parking_fg.add_to(m)
                    recommendations_fg.add_to(m)
                    folium.LayerControl(collapsed=False).add_to(m)
                    note_html = '''
                    <div style="position: fixed; 
                         top: 10px; right: 10px; width: 200px; height: auto;
                         font-size:12px; font-family: 'Arial'; z-index:9998; 
                         background-color: rgba(255, 255, 255, 0.8); padding: 8px; border-radius: 5px; border:1px solid grey;">
                         <b>Note:</b> Map view is restricted to California area.
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(note_html))
                    if 'recommendation_status' in top_recommendations.columns:
                        legend_html = '''
                        <div style="position: fixed; 
                             bottom: 50px; left: 10px; width: 180px; height: auto;
                             font-size:12px; font-family: 'Arial'; z-index:9999; 
                             background-color: white; padding: 10px; border-radius: 5px; border:1px solid grey;">
                             <p style="margin: 0 0 5px 0; font-weight: bold;">Recommendation Status</p>
                             <div style="display: flex; align-items: center; margin-bottom: 4px;">
                                 <span style="display: inline-block; height: 15px; width: 15px; background-color: #FF0000; margin-right: 5px; border-radius: 50%;"></span>
                                 <span>Highly Recommended</span>
                             </div>
                             <div style="display: flex; align-items: center; margin-bottom: 4px;">
                                 <span style="display: inline-block; height: 15px; width: 15px; background-color: #FF8C00; margin-right: 5px; border-radius: 50%;"></span>
                                 <span>Recommended</span>
                             </div>
                             <div style="display: flex; align-items: center;">
                                 <span style="display: inline-block; height: 15px; width: 15px; background-color: #4682B4; margin-right: 5px; border-radius: 50%;"></span>
                                 <span>Not Recommended</span>
                             </div>
                        </div>
                        '''
                        m.get_root().html.add_child(folium.Element(legend_html))
                    title_text = f"Top {len(top_recommendations)} Recommended Locations " + (f"in ZIP {zip_filter}" if zip_filter else "Across All ZIP Codes")
                    st.subheader(title_text)
                    folium_static(m, width=map_width, height=map_height)
                    if ml_explanation:
                        components.html(ml_explanation, height=700, scrolling=True)
                    
                    tab1_sub, tab2_sub = st.tabs(["Summary View", "Detailed Data"])
                    with tab1_sub:
                        st.subheader("Key Metrics for Recommended Locations")
                        summary_cols = ['osmid', 'zip', 'new_demand_score','distance_score']
                        if 'recommendation_status' in top_recommendations.columns:
                            summary_cols.append('recommendation_status')
                        summary_cols.extend(['distance_to_nearest_charger', 'chargers_in_radius', 'traffic'])
                        available_summary_cols = [col for col in summary_cols if col in top_recommendations.columns]
                        summary_df = top_recommendations[available_summary_cols].copy()
                        summary_df['rank'] = range(1, len(summary_df) + 1)
                        summary_df = summary_df.set_index('rank')
                        rename_dict = {
                            'osmid': 'ID', 
                            'zip': 'ZIP', 
                            'zone_type': 'Zone Type',
                            'new_demand_score': 'Demand Score',
                            'recommendation_status': 'Status',
                            'distance_to_nearest_charger': 'Distance to Nearest (m)',
                            'chargers_in_radius': 'Nearby Chargers',
                            'traffic': 'Traffic Volume'
                        }
                        summary_df = summary_df.rename(columns={k: v for k, v in rename_dict.items() if k in summary_df.columns})
                        if 'Status' in summary_df.columns:
                            def status_color(val):
                                if val == 'Highly Recommended':
                                    return 'background-color: #ffcccc'
                                elif val == 'Recommended':
                                    return 'background-color: #ffedcc'
                                else:
                                    return 'background-color: #cce6ff'
                            st.dataframe(summary_df.style
                                        .format(formatter={
                                            'Demand Score': '{:.3f}',
                                            'Distance to Nearest (m)': '{:.0f}',
                                            'Traffic Volume': '{:.0f}'
                                        })
                                        .applymap(status_color, subset=['Status']))
                        else:
                            st.dataframe(summary_df.style.format(formatter={
                                'Demand Score': '{:.3f}',
                                'Distance to Nearest (m)': '{:.0f}',
                                'Traffic Volume': '{:.0f}'
                            }))
                    with tab2_sub:
                        st.subheader("Complete Data for Recommended Locations")
                        display_cols = [col for col in top_recommendations.columns if col != 'geometry']
                        full_df = top_recommendations[display_cols].copy()
                        full_df['rank'] = range(1, len(full_df) + 1)
                        full_df = full_df.set_index('rank')
                        st.dataframe(full_df.style.format(formatter={col: '{:.3f}' for col in full_df.select_dtypes('float').columns}))
                    
                    if st.sidebar.checkbox("Show Factor Analysis", value=True) and 'recommendation_status' in top_recommendations.columns:
                        st.subheader("Factor Contribution Analysis")
                        score_columns = [
                            'distance_score', 
                            'radius_score', 
                            'cs_total_score', 
                            'traffic_score', 
                            'population_score', 
                            'income_score', 
                            'zoning_score'
                        ]
                        available_score_cols = [col for col in score_columns if col in top_recommendations.columns]
                        if len(available_score_cols) > 0:
                            col1_heat, col2_radar = st.columns(2)
                            with col1_heat:
                                st.subheader("Factor Score Heatmap")
                                heatmap_data = top_recommendations[available_score_cols].head(10).copy()
                                column_labels = {
                                    'distance_score': 'Distance (0.5)', 
                                    'radius_score': 'Nearby Chargers (-0.3)', 
                                    'cs_total_score': 'Charging Points (-0.2)', 
                                    'traffic_score': 'Traffic (0.1)', 
                                    'population_score': 'Population (0.2)', 
                                    'income_score': 'Income (0.1)',
                                    'zoning_score': 'Zoning (0.1)'
                                }
                                heatmap_data.columns = [column_labels.get(col, col) for col in heatmap_data.columns]
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f', linewidths=.5, ax=ax)
                                plt.title('Factor Contribution Scores by Location')
                                plt.ylabel('Location Rank')
                                plt.tight_layout()
                                st.pyplot(fig)
                                # Dynamically generate the factor weights text based on chosen_weights
                                weight_text = "**Factor Weights (" + ("PCA Optimized" if weight_mode=="PCA Optimized" else "Manual") + "):**\n"
                                for feat, w in chosen_weights.items():
                                    friendly = feat.replace('_score', '').replace('_', ' ').title()
                                    weight_text += f"- **{friendly}** ({w:.2f})\n"
                                st.markdown(weight_text)
                            with col2_radar:
                                st.subheader("Top Location Profile")
                                plot_radar_in_streamlit(top_recommendations, available_score_cols)
                    
                    col1_dl, col2_dl = st.columns(2)
                    with col1_dl:
                        csv = full_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name="ev_recommendations.csv",
                            mime="text/csv",
                        )
                    with col2_dl:
                        if st.button("Download Map as HTML"):
                            map_html = m._repr_html_()
                            with open("ev_demand_map.html", "w") as f:
                                f.write(map_html)
                            st.success("Map saved as ev_demand_map.html")
                            with open("ev_demand_map.html", "rb") as f:
                                st.download_button(
                                    label="Click to Download Map",
                                    data=f,
                                    file_name="ev_demand_map.html",
                                    mime="text/html"
                                )
            else:
                st.info("Use the filters in the sidebar and select a weight mode, then click 'Show Recommendations' to generate the map.")
                st.markdown("""
                ### How the Machine Learning Analysis Works
                When you use the ML Analysis option, the app will:
                1. **Cluster Locations**: Group similar locations based on their characteristics using K-means clustering.
                2. **Define Thresholds**: Automatically determine appropriate thresholds for each factor.
                3. **Generate Natural Language**: Translate data points into human-readable explanations.
                4. **Determine Recommendations**: Classify locations as "Highly Recommended," "Recommended," or "Not Recommended."
                5. **Create Visualizations**: Show factor contributions through heatmaps and radar charts.
                The ML analysis helps identify patterns that might not be obvious from the raw scores alone.
                """)
    except Exception as e:
        st.error(f"Error loading supporting data: {e}")
        st.info("Please ensure your supporting data files exist and are properly formatted.")
        st.exception(e)
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please ensure your main data is properly loaded and contains the required columns.")
    st.exception(e)

# ---------------------------------------------------------------------
# Weight Optimization & Analysis UI Component
# ---------------------------------------------------------------------
st.sidebar.header("Weight Optimization & Analysis")
render_weight_analysis_ui_1(merged_features, WeightOptimizer_1)
