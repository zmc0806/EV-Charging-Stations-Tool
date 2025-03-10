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

# Set page title and layout
st.set_page_config(
    page_title="EV Charging Station Demand Map",
    layout="wide"
)

# Page title and description
st.title("EV Charging Station Demand Score Map with ML Analysis")
st.markdown("This map helps identify optimal locations for EV charging stations using machine learning and geospatial analysis.")

# Function to map zoning categories
def map_zoning_category(zone_code):
    """Map zone codes to categories"""
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
    
    if isinstance(zone_code, str):  # Check if zone_code is a string
        for category, prefixes in zoning_categories.items():
            if any(zone_code.startswith(prefix) for prefix in prefixes):
                return category
    return 'Other'  # Return 'Other' for NaN or non-string values

# Function to load data
@st.cache_data
def load_data():
    """Load and prepare the geodataframe with merged features"""
    # Replace with your actual data loading code
    merged_features = pd.read_csv("features.csv")
    merged_features['geometry'] = gpd.GeoSeries.from_wkt(merged_features['geometry'])
    merged_features = gpd.GeoDataFrame(merged_features, geometry='geometry')
    
    # Ensure the correct coordinate reference system is set
    merged_features.set_crs(epsg=4326, inplace=True)
    
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
    
    # Manually assign weights based on expert knowledge
    merged_features['distance_score'] = merged_features['percentile_distance_to_nearest_charger'] * 0.5
    merged_features['radius_score'] = merged_features['percentile_chargers_in_radius'] * -0.3
    merged_features['cs_total_score'] = merged_features['percentile_cs_total'] * -0.2
    merged_features['traffic_score'] = merged_features['percentile_traffic'] * 0.1
    merged_features['population_score'] = merged_features['percentile_Population'] * 0.2
    merged_features['income_score'] = merged_features['percentile_Median Income'] * 0.1
    
    # Calculate combined demand score including traffic score
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
    """Load zoning data, parking data, and other supporting datasets"""
    # Load zoning data
    zoning_data = gpd.read_file("zoning_datasd.geojson")
    zoning_data = zoning_data.to_crs(epsg=4326)
    
    # Process zone_type if not already present
    if 'zone_type' not in zoning_data.columns and 'zone_name' in zoning_data.columns:
        zoning_data['zone_type'] = zoning_data['zone_name'].apply(map_zoning_category)
    
    # Load public parking data
    public_parking_gdf = gpd.read_file("parking_data.geojson")
    public_parking_gdf = public_parking_gdf.to_crs(epsg=4326)
    
    # Load lots with zone data
    lots = gpd.read_file("lots.geojson")
    lots = lots.to_crs(epsg=4326)
    
    return zoning_data, public_parking_gdf, lots

# Function to analyze scores with ML and define thresholds
@st.cache_data
def analyze_scores_with_ml(_data):
    """
    Use machine learning techniques to analyze scores, determine thresholds, and classify locations.
    
    Parameters:
    _data (DataFrame): DataFrame containing all locations and their scores.
    
    Returns:
    dict: Dictionary containing thresholds and clustering information.
    """
    # Work with a copy of the data
    data = _data.copy()
    
    # Select relevant features for clustering
    score_columns = [col for col in data.columns if col.endswith('_score') and col != 'demand_score']

    if not score_columns:
        # If no score columns are found, use default values
        return {
            'demand_thresholds': [0.33, 0.66],  # Default thresholds
            'factor_thresholds': {},
            'clusters': None,
            'cluster_descriptions': {}
        }
    
    # Extract features for clustering
    features = data[score_columns].copy()
    features = features.fillna(0)  # Fill NaN values
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Determine the optimal number of clusters (simplified version)
    n_clusters = min(5, len(data) // 10) if len(data) > 10 else 3
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add clustering assignment to the data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Compute average scores for each cluster
    cluster_profiles = data_with_clusters.groupby('cluster')[score_columns + ['demand_score']].mean()
    
    # Sort clusters by demand score (from high to low)
    cluster_profiles = cluster_profiles.sort_values('demand_score', ascending=False)
    
    # Assign descriptive labels to clusters
    cluster_descriptions = {}
    for i, (cluster_idx, profile) in enumerate(cluster_profiles.iterrows()):
        if i == 0:
            desc = "High Priority"
        elif i == 1 and n_clusters > 2:
            desc = "Medium Priority"
        else:
            desc = "Low Priority"
        
        # Find the top 3 contributing factors
        factor_scores = profile[score_columns].sort_values(ascending=False)
        top_factors = factor_scores.index[:3].tolist()
        factor_desc = [f.replace('_score', '').replace('_', ' ').title() for f in top_factors]
        
        cluster_descriptions[cluster_idx] = {
            'description': desc,
            'top_factors': factor_desc,
            'avg_demand_score': profile['demand_score']
        }
    
    # Determine thresholds for demand score
    demand_scores = data['demand_score'].dropna()
    if len(demand_scores) > 0:
        demand_thresholds = [
            demand_scores.quantile(0.33),
            demand_scores.quantile(0.66)
        ]
    else:
        demand_thresholds = [0.33, 0.66]  # Default fallback values
    
    # Determine threshold values for each factor
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
        'clusters': clusters.tolist(),  # Convert numpy array to list for serialization
        'cluster_descriptions': cluster_descriptions
    }

# Function to generate HTML explanation text for recommendations based on ML analysis
def generate_ml_explanations(recommendations_df, ml_analysis):
    """
    Generate a human-readable explanation of the recommended locations based on machine learning analysis.
    
    Parameters:
    recommendations_df (DataFrame): DataFrame containing recommended locations and their scores.
    ml_analysis (dict): ML analysis results, including thresholds and clustering.
    
    Returns:
    str: HTML formatted explanation text, along with the updated recommendations DataFrame.
    """
    # Get thresholds and clustering information
    demand_thresholds = ml_analysis['demand_thresholds']
    factor_thresholds = ml_analysis.get('factor_thresholds', {})
    cluster_descriptions = ml_analysis.get('cluster_descriptions', {})
    
    # Record recommendation counts
    rec_counts = {
        "Highly Recommended": 0,
        "Recommended": 0,
        "Not Recommended": 0
    }
    
    # Add recommendation status for each location
    recommendations_with_status = recommendations_df.copy()
    recommendations_with_status['recommendation_status'] = recommendations_with_status.apply(
        lambda row: get_recommendation_status(
            row['demand_score'], 
            demand_thresholds,
            cluster_descriptions.get(row.get('cluster'))
        ),
        axis=1
    )
    
    # Count by status
    status_counts = recommendations_with_status['recommendation_status'].value_counts()
    for status, count in status_counts.items():
        if status in rec_counts:
            rec_counts[status] = count
    
    # Get top recommendation
    top_rec = recommendations_with_status.iloc[0]
    
    # Identify score columns
    score_columns = [col for col in recommendations_df.columns if col.endswith('_score') and col != 'demand_score']
    
    # Build explanation text
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
    
    # Add detailed analysis for the top 3 recommended locations
    for i, (idx, row) in enumerate(recommendations_with_status.head(3).iterrows()):
        explanation_html += f"""
            <li style="margin-bottom: 1em;"><strong>Rank #{i+1} (ID: {row.osmid}):</strong> {row.recommendation_status} - 
            Demand score: {row.demand_score:.2f}
        """
        
        # Add description of the zoning type
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
        
        # Add clustering information
        if 'cluster' in row and row['cluster'] in cluster_descriptions:
            cluster_info = cluster_descriptions[row['cluster']]
            explanation_html += f"""
                <br>Part of {cluster_info['description']} cluster based on {', '.join(cluster_info['top_factors'][:2])}
            """
        
        explanation_html += "<ul style='margin-top: 0.5em;'>"
        
        # Add detailed description for each factor
        # Distance to nearest charger
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
        
        # Number of chargers in radius
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
        
        # Traffic volume
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
        
        # Population density
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
        
        # Income level
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
        
        # Add recommendation for the specific location
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
    
    # Add overall ML-based analysis
    explanation_html += """</ul>
        
        <p><strong>Machine Learning Insights:</strong></p>
        <ul style="margin-top: 0.5em;">
    """
    
    # Add clustering insights
    if cluster_descriptions:
        n_clusters = len(cluster_descriptions)
        explanation_html += f"""
            <li>Our machine learning algorithm identified {n_clusters} distinct location profiles based on multiple factors.</li>
        """
        
        # Add top cluster description
        top_cluster_idx = next(iter(cluster_descriptions))
        top_cluster = cluster_descriptions[top_cluster_idx]
        explanation_html += f"""
            <li>The highest priority cluster shows strong potential with average demand score of 
            {top_cluster['avg_demand_score']:.2f}, characterized by {', '.join(top_cluster['top_factors'][:2])}.</li>
        """
    
    # Add threshold insights
    explanation_html += f"""
        <li>Locations with demand scores above {demand_thresholds[1]:.2f} are considered high priority 
        and below {demand_thresholds[0]:.2f} are low priority based on our quantile analysis.</li>
    """
    
    # The most influential factors across all recommendations
    if score_columns:
        avg_scores = recommendations_df[score_columns].mean().sort_values(ascending=False)
        top_factors = avg_scores.index[:3]
        factor_descs = [col.replace('_score', '').replace('_', ' ').title() for col in top_factors]
        explanation_html += f"""
            <li>The most influential factors across all recommendations are {', '.join(factor_descs)}.</li>
        """
    
    # Add insights about key determining factors
    explanation_html += """
        <li>Distance from existing charging infrastructure is the strongest predictor of need, with a weight of 0.5 in our model.</li>
        <li>Population density (weight: 0.2) and nearby charger saturation (weight: -0.3) are also key determinants.</li>
    """
    
    # Conclusion and recommendation summary
    explanation_html += """</ul>
        
        <p><strong>Recommendation Summary:</strong> Based on our machine learning analysis, we recommend prioritizing 
        the highly recommended locations for immediate consideration, followed by the recommended locations as secondary options. 
        Locations marked as not recommended should be excluded from immediate planning.</p>
        
        <p><em>Methodology: Our model uses a weighted sum of factors including distance to nearest charger (0.5), 
        nearby chargers (-0.3), existing charging points (-0.2), traffic volume (0.1), population density (0.2), 
        income level (0.1), and zoning suitability (0.1), combined with K-means clustering to identify patterns.</em></p>
    </div>
    """
    
    return explanation_html, recommendations_with_status

# Function to determine recommendation status based on demand score and clustering information
def get_recommendation_status(demand_score, thresholds, cluster_desc=None):
    """
    Determine recommendation status based on demand score and clustering information.
    
    Parameters:
    demand_score (float): The demand score.
    thresholds (list): List of thresholds [low_threshold, high_threshold].
    cluster_desc (dict, optional): Clustering description information.
    
    Returns:
    str: Recommendation status.
    """
    # Basic threshold-based determination
    if demand_score < thresholds[0]:
        status = "Not Recommended"
    elif demand_score > thresholds[1]:
        status = "Highly Recommended"
    else:
        status = "Recommended"
    
    # Override if clustering information is available
    if cluster_desc:
        if cluster_desc['description'] == "High Priority":
            status = "Highly Recommended"
        elif cluster_desc['description'] == "Low Priority":
            if status == "Highly Recommended":
                status = "Recommended"  # Downgrade based on clustering
    
    return status

# Function to create the combined map
def create_combined_map(zoning_data, public_parking_gdf, lots, merged_features):
    """Create a combined map with all layers"""

    def convert_timestamps(gdf):
        for col in gdf.columns:
            if gdf[col].dtype == 'datetime64[ns]' or hasattr(gdf[col], 'dt'):
                gdf[col] = gdf[col].astype(str)
        return gdf
    zoning_data = convert_timestamps(zoning_data)
    public_parking_gdf = convert_timestamps(public_parking_gdf)
    lots = convert_timestamps(lots)
    merged_features = convert_timestamps(merged_features)
    # Create a base map centered on San Diego
    combined_map = folium.Map(location=[32.7157, -117.1611], zoom_start=12, tiles="cartodbpositron")
    
    # Define color mapping for zoning types
    color_mapping = {
        'Commercial': '#4A90E2',       # Brighter blue
        'Office': '#50C878',           # Emerald green
        'Residential High': '#E74C3C', # Bright red
        'Residential Medium': '#F4D03F', # Golden yellow
        'Residential Low': '#F9E79F',  # Light yellow
        'Residential Mixed': '#FF69B4', # Bright pink
        'Industrial': '#95A5A6',       # Silver gray
        'Mixed Use': '#2CCBCB',        # Blue-green
        'Agricultural': '#DAA520',     # Goldenrod
        'Open Space': '#87CEFA',       # Light blue
        'Planned': '#00CED1',          # Teal
        'Transit': '#9370DB',          # Medium purple
        'Other': '#5D6D7E',            # Dark gray
        'Multiple': '#9932CC'          # Dark orchid
    }
    
    # Function: Style for zoning polygons with improved error handling
    def zoning_style(feature):
        properties = feature['properties']
        
        # Check for different possible field names for the zone type
        if 'zone_type' in properties:
            zone_type = properties['zone_type']
        elif 'zone_name' in properties:
            # Map zone_name to zone_type if that's what's available
            zone_name = properties['zone_name']
            zone_type = map_zoning_category(zone_name)
        else:
            # Default if neither field exists
            zone_type = 'Other'
        
        color = color_mapping.get(zone_type, 'gray')
        return {'fillColor': color, 'color': 'black', 'weight': 1, 'fillOpacity': 0.3}
    
    # Function: Get coordinates from geometry
    def get_coords(geometry):
        if isinstance(geometry, (Point, Polygon, MultiPolygon)):
            centroid = geometry.centroid
            return centroid.y, centroid.x
        return np.nan, np.nan
    
    # Create layer groups
    zoning_layer = folium.FeatureGroup(name='Zoning Areas')
    parking_layer = folium.FeatureGroup(name='Public Parking Lots')
    zoned_parking_layer = folium.FeatureGroup(name='Parking by Zone Type')
    charger_distance_layer = folium.FeatureGroup(name='Distance to Nearest Charger')
    
    # Debug: Print the first feature's properties to see the available fields
    st.sidebar.write("Data Loading Status:")
    
    if len(zoning_data) > 0:
        st.sidebar.success(f"✅ Zoning data loaded: {len(zoning_data)} features")
        first_feature = zoning_data.iloc[0]
        
        # Print to debug
        debug_info = {}
        for col in zoning_data.columns:
            debug_info[col] = str(first_feature[col])[:50]  # Show first 50 chars
        
        if st.sidebar.checkbox("Show zoning data details", False):
            st.sidebar.write("First zoning feature properties:", debug_info)
    else:
        st.sidebar.error("❌ Zoning data is empty")
    
    # 1. Add zoning layer with improved error handling
    try:
        # First convert to GeoJSON format for folium
        zoning_geojson = zoning_data.__geo_interface__
        
        folium.GeoJson(
            zoning_geojson,
            style_function=zoning_style,
            name='Zoning Areas'
        ).add_to(zoning_layer)
    except Exception as e:
        st.sidebar.error(f"Error adding zoning layer: {e}")
    
    # 2. Add original parking data
    try:
        for _, row in public_parking_gdf.iterrows():
            if row.geometry.geom_type == "Point":
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color="#004080",  # Dark blue border
                    fill=True,
                    fill_color="#0066CC",  # More saturated blue fill
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
    
    # 3. Add parking data with zone type information
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
    
    # 4. Add charger distance information layer
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
    
    # 5. Add zoning legend
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
    
    # 6. Add second legend for the charger distance layer
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
    
    # Add layers to the map in order (bottom to top)
    zoning_layer.add_to(combined_map)
    parking_layer.add_to(combined_map)
    zoned_parking_layer.add_to(combined_map)
    charger_distance_layer.add_to(combined_map)
    
    # Add measurement tools
    MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(combined_map)
    
    # Add layer control
    folium.LayerControl(position='topright', collapsed=True).add_to(combined_map)
    
    # Add custom layer control for better UX
    custom_layer_control = '''
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        // Wait for the map to finish loading
        setTimeout(function() {
            // Add additional click handling for the layer control
            var checkboxes = document.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]');
            
            if (checkboxes.length >= 4) {
                // By default, turn off some layers
                checkboxes[1].click(); // Turn off Public Parking Lot layer
                checkboxes[2].click(); // Turn off Zoned Parking Lot layer
                
                // Add click event handling
                checkboxes.forEach(function(checkbox, index) {
                    checkbox.addEventListener('change', function() {
                        // Record the current layer state
                        console.log("Layer " + index + " is now: " + (this.checked ? "on" : "off"));
                        
                        // If the user opens multiple conflicting layers, show a tip
                        var enabledLayers = Array.from(checkboxes).filter(cb => cb.checked).length;
                        if (enabledLayers > 2) {
                            // Create or update the tip element
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
                            
                            // Hide the tip after 3 seconds
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
    
    # Add title at the top
    title_html = '''
    <h3 align="center" style="font-size:20px; font-family: 'Arial'; margin-top: 10px;">
        <b>EV Charging Station Geographic Context</b>
    </h3>
    '''
    combined_map.get_root().html.add_child(folium.Element(title_html))
    
    return combined_map

# Function to create a radar chart for factor analysis of the top location
def create_radar_chart(top_recommendations, score_columns):
    """
    Create a radar chart for factor analysis of the top location.
    
    Parameters:
    top_recommendations (DataFrame): DataFrame containing the recommended locations.
    score_columns (list): List of score column names.
    
    Returns:
    matplotlib.figure.Figure: Radar chart figure object.
    str: Error message (if any).
    """
    # Get data for the top location
    if len(top_recommendations) == 0:
        return None, "No recommendations data available"
    
    top_loc = top_recommendations.iloc[0]
    
    # Ensure that only score columns present in the data are used
    available_score_cols = [col for col in score_columns if col in top_loc.index]
    
    # A radar chart requires at least 3 dimensions to be meaningful
    if len(available_score_cols) < 3:
        return None, "Not enough score dimensions for radar chart (need at least 3)"
    
    # Obtain radar chart data
    radar_data = top_loc[available_score_cols].copy()
    
    # Check data types to ensure numeric operations are possible
    for col in available_score_cols:
        if not pd.api.types.is_numeric_dtype(radar_data[col]):
            try:
                radar_data[col] = pd.to_numeric(radar_data[col])
            except:
                return None, f"Column {col} contains non-numeric data"
    
    # Normalize values to between -1 and 1 for better visualization
    max_abs = max(abs(radar_data.min()), abs(radar_data.max()))
    if max_abs > 0:
        radar_data = radar_data / max_abs
    
    # Create radar chart data
    # Using friendly labels for readability
    column_labels = {
        'distance_score': 'Distance (0.5)', 
        'radius_score': 'Nearby Chargers (-0.3)', 
        'cs_total_score': 'Charging Points (-0.2)', 
        'traffic_score': 'Traffic (0.1)', 
        'population_score': 'Population (0.2)', 
        'income_score': 'Income (0.1)',
        'zoning_score': 'Zoning (0.1)'
    }
    
    categories = [column_labels.get(col, col.replace('_score', '').replace('_', ' ').title()) 
                 for col in available_score_cols]
    values = radar_data.tolist()
    
    # To close the radar chart, append the first value at the end
    values.append(values[0])
    categories.append(categories[0])
    
    # Calculate angles for each category (evenly distributed around the circle)
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=True)
    
    # Ensure that angles and values arrays have the same length
    if len(angles) != len(values):
        return None, f"Dimension mismatch: angles ({len(angles)}) and values ({len(values)}) have different lengths"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    
    # Set category labels (excluding the repeated last one)
    plt.xticks(angles[:-1], categories[:-1])
    
    # Add background grid
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
    ax.grid(True)
    
    # Add title
    plt.title(f'Factor Profile for Top Location (ID: {top_loc.osmid})', pad=20)
    
    return fig, None

# Safely display radar chart in Streamlit
def plot_radar_in_streamlit(top_recommendations, score_columns):
    """
    Safely create and display a radar chart in Streamlit.
    
    Parameters:
    top_recommendations (DataFrame): DataFrame containing recommended locations.
    score_columns (list): List of score column names.
    """
    try:
        # Create radar chart with error handling
        fig, error = create_radar_chart(top_recommendations, score_columns)
        
        if error:
            st.warning(f"Could not create radar chart: {error}")
            return
            
        if fig:
            st.pyplot(fig)
            
            # Add analysis for the top location
            top_loc = top_recommendations.iloc[0]
            
            # Identify basic location info
            st.markdown(f"""
            **ID**: {top_loc.osmid} | **ZIP**: {top_loc.zip} | **Zone**: {top_loc.zone_type}
            """)
            
            # Identify strongest factor
            scores = {col: top_loc[col] for col in score_columns if col in top_loc}
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_scores:
                strongest = sorted_scores[0]
                strongest_factor = strongest[0].replace('_score', '').replace('_', ' ').title()
                
                # Provide specific descriptions based on factor type
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
                
                # If there are multiple factors, also show the weakest
                if len(sorted_scores) > 1:
                    weakest = sorted_scores[-1]
                    weakest_factor = weakest[0].replace('_score', '').replace('_', ' ').title()
                    weakest_desc = factor_descriptions.get(weakest[0], weakest_factor)
                    st.markdown(f"**Lowest Factor**: {weakest_factor} ({weakest[1]:.2f}) - {weakest_desc}")
        else:
            st.info("Not enough data to create radar chart")
    except Exception as e:
        st.warning(f"Error creating radar chart: {e}")

# Main application
try:
    # Load data
    merged_features = load_data()
    
    try:
        # Load supporting data
        zoning_data, public_parking_gdf, lots = load_supporting_data()
        
        # Get available ZIP codes for the dropdown
        available_zips = sorted(merged_features['zip'].unique())
        
        # Create tabs for different map views
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
            
            # Create the combined map
            combined_map = create_combined_map(zoning_data, public_parking_gdf, lots, merged_features)
            
            # Display the combined map
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
            
            # Sidebar for filters
            st.sidebar.header("Filter Options")
            
            # ZIP code filter
            zip_options = ["All ZIP Codes"] + [str(z) for z in available_zips]
            selected_zip = st.sidebar.selectbox("Select ZIP Code:", zip_options)
            
            # Convert "All ZIP Codes" to None for filtering
            zip_filter = None if selected_zip == "All ZIP Codes" else int(selected_zip)
            
            # Number of recommendations slider
            num_recommendations = st.sidebar.slider(
                "Number of Recommendations:", 
                min_value=1, 
                max_value=10, 
                value=2
            )
            
            focus_on_selection = True
            focus_buffer = 0.05
            map_width = 800
            map_height = 500
            
            # Show advanced ML options
            st.sidebar.header("Machine Learning Options")
            use_ml_analysis = st.sidebar.checkbox("Use Machine Learning Analysis", value=True)
            
            # Create map button
            if st.sidebar.button("Show Recommendations"):
                # Show a spinner while preparing the map and running ML analysis...
                with st.spinner("Preparing map and running ML analysis..."):
                    # Filter data
                    if zip_filter:
                        filtered_data = merged_features[merged_features['zip'] == zip_filter]
                    else:
                        filtered_data = merged_features
                    
                    # Sort by demand score and get top recommendations
                    top_recommendations = filtered_data.sort_values('demand_score', ascending=False).head(num_recommendations)
                    
                    # Run ML analysis on the data
                    if use_ml_analysis:
                        ml_analysis = analyze_scores_with_ml(filtered_data)
                        ml_explanation, top_recommendations = generate_ml_explanations(top_recommendations, ml_analysis)
                    else:
                        ml_analysis = None
                        ml_explanation = None
                    
                    # Determine map center and bounds
                    if focus_on_selection and len(top_recommendations) > 0:
                        # Calculate the bounds of the recommendations
                        if all(hasattr(geom, 'y') and hasattr(geom, 'x') for geom in top_recommendations.geometry):
                            # For point geometries
                            min_lat = top_recommendations.geometry.y.min() - focus_buffer
                            max_lat = top_recommendations.geometry.y.max() + focus_buffer
                            min_lon = top_recommendations.geometry.x.min() - focus_buffer
                            max_lon = top_recommendations.geometry.x.max() + focus_buffer
                            
                            # Center the map on the recommendations
                            center_lat = (min_lat + max_lat) / 2
                            center_lon = (min_lon + max_lon) / 2
                            map_center = [center_lat, center_lon]
                            
                            # Calculate appropriate zoom level (bounds will be used for fit_bounds later)
                            sw = [min_lat, min_lon]
                            ne = [max_lat, max_lon]
                            
                        else:
                            # For mixed or non-point geometries, use centroids
                            centroids = top_recommendations.geometry.centroid
                            min_lat = centroids.y.min() - focus_buffer
                            max_lat = centroids.y.max() + focus_buffer
                            min_lon = centroids.x.min() - focus_buffer
                            max_lon = centroids.x.max() + focus_buffer
                            
                            center_lat = (min_lat + max_lat) / 2
                            center_lon = (min_lon + max_lon) / 2
                            map_center = [center_lat, center_lon]
                            
                            sw = [min_lat, min_lon]
                            ne = [max_lat, max_lon]
                    else:
                        # Default to San Diego center
                        map_center = [32.7157, -117.1611]
                        sw, ne = None, None
                    
                    # Define California bounds (approximate)
                    california_bounds = [
                        [32.5, -124.5],  # Southwest corner
                        [42.0, -114.0]   # Northeast corner
                    ]
                    
                    # Create Folium map with zoom restrictions
                    m = folium.Map(
                        location=map_center,
                        zoom_start=12,
                        tiles='CartoDB positron',
                        min_zoom=7,  # Restrict minimum zoom level to roughly California state level
                        max_zoom=18,
                        zoom_control=True,
                        scrollWheelZoom=True,
                        dragging=True
                    )
                    
                    # Add California bounds as a rectangle for context (optional)
                    folium.Rectangle(
                        bounds=california_bounds,
                        color='gray',
                        weight=1,
                        fill=False,
                        opacity=0.3,
                        tooltip="California state bounds (approximate)"
                    ).add_to(m)
                    
                    # If we have bounds, fit the map to those bounds
                    if focus_on_selection and sw and ne:
                        m.fit_bounds([sw, ne])
                    
                    # Add JavaScript to restrict the maximum bounds so the user cannot pan outside California
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
                    
                    # Create color scale
                    colormap = cm.linear.YlOrRd_09.scale(filtered_data['demand_score'].min(), filtered_data['demand_score'].max())
                    colormap.caption = "Demand Score"
                    colormap.add_to(m)
                    
                    # Add all parking lots layer (initially hidden)
                    all_parking_fg = folium.FeatureGroup(name="All Parking Lots", show=False)
                    
                    # If focusing on selection, only show parking lots in the visible area (with a buffer)
                    if focus_on_selection and sw and ne:
                        visible_data = filtered_data[
                            (filtered_data.geometry.centroid.y >= min_lat - 0.1) & 
                            (filtered_data.geometry.centroid.y <= max_lat + 0.1) & 
                            (filtered_data.geometry.centroid.x >= min_lon - 0.1) & 
                            (filtered_data.geometry.centroid.x <= max_lon + 0.1)
                        ]
                    else:
                        visible_data = filtered_data
                    
                    # Add points to the all parking lots layer
                    for _, row in visible_data.iterrows():
                        color = colormap(row['demand_score'])
                        tooltip = f"ID: {row.osmid}, Score: {row.demand_score:.2f}, ZIP: {row.zip}"
                        
                        if hasattr(row.geometry, 'y') and hasattr(row.geometry, 'x'):
                            # Point geometry
                            folium.Circle(
                                location=(row.geometry.y, row.geometry.x),
                                radius=15,
                                color=color,
                                fill=True,
                                fill_opacity=0.6,
                                tooltip=tooltip
                            ).add_to(all_parking_fg)
                        else:
                            # Attempt to add polygon or other geometry
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
                    
                    # Add recommendations layer
                    recommendations_fg = folium.FeatureGroup(name="Recommended Parking Lots", show=True)
                    
                    # Add recommended points with highlight colors based on recommendation status
                    for i, (_, row) in enumerate(top_recommendations.iterrows()):
                        # Determine color based on recommendation status if available
                        if 'recommendation_status' in row:
                            if row['recommendation_status'] == "Highly Recommended":
                                highlight_color = '#FF0000'  # Bright red for highly recommended
                            elif row['recommendation_status'] == "Recommended":
                                highlight_color = '#FF8C00'  # Orange for recommended
                            else:
                                highlight_color = '#4682B4'  # Blue for not recommended
                        else:
                            highlight_color = '#FF4500'  # Default orange-red
                        
                        # Create tooltip with recommendation status if available
                        tooltip_content = [
                            f"Rank #{i+1}",
                            f"ID: {row.osmid}",
                            f"Demand Score: {row.demand_score:.2f}",
                            f"ZIP: {row.zip}",
                            f"Zone Type: {row.zone_type if 'zone_type' in row else 'N/A'}"
                        ]
                        
                        if 'recommendation_status' in row:
                            tooltip_content.insert(1, f"Status: {row.recommendation_status}")
                        
                        tooltip = "<br>".join(tooltip_content)
                        
                        if hasattr(row.geometry, 'y') and hasattr(row.geometry, 'x'):
                            # Point geometry
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
                            # Attempt to add polygon or other geometry
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
                    
                    # Add layers to map
                    all_parking_fg.add_to(m)
                    recommendations_fg.add_to(m)
                    
                    # Add layer control and keep it expanded
                    folium.LayerControl(collapsed=False).add_to(m)
                    
                    # Add a note about restricted view
                    note_html = '''
                    <div style="position: fixed; 
                         top: 10px; right: 10px; width: 200px; height: auto;
                         font-size:12px; font-family: 'Arial'; z-index:9998; 
                         background-color: rgba(255, 255, 255, 0.8); padding: 8px; border-radius: 5px; border:1px solid grey;">
                         <b>Note:</b> Map view is restricted to California area.
                    </div>
                    '''
                    m.get_root().html.add_child(folium.Element(note_html))
                    
                    # Add legend for recommendation status
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
                    
                    # Set map title
                    title_text = f"Top {len(top_recommendations)} Recommended Locations " + (f"in ZIP {zip_filter}" if zip_filter else "Across All ZIP Codes")
                    
                    # Display the map in Streamlit with user-defined dimensions
                    st.subheader(title_text)
                    folium_static(m, width=map_width, height=map_height)
                    
                    # Display ML-based explanation with components for better HTML rendering
                    if ml_explanation:
                        components.html(ml_explanation, height=700, scrolling=True)
                    
                    # Create tabs for different data views
                    tab1, tab2 = st.tabs(["Summary View", "Detailed Data"])
                    
                    with tab1:
                        # Display a more focused summary table with recommendation status
                        st.subheader("Key Metrics for Recommended Locations")
                        summary_cols = ['osmid', 'zip', 'demand_score','distance_score']
                        
                        if 'recommendation_status' in top_recommendations.columns:
                            summary_cols.append('recommendation_status')
                            
                        summary_cols.extend(['distance_to_nearest_charger', 'chargers_in_radius', 'traffic'])
                        
                        # Only include columns that exist in the data
                        available_summary_cols = [col for col in summary_cols if col in top_recommendations.columns]
                        
                        summary_df = top_recommendations[available_summary_cols].copy()
                        summary_df['rank'] = range(1, len(summary_df) + 1)
                        summary_df = summary_df.set_index('rank')
                        
                        # Rename columns for readability
                        # Rename columns for readability
                        rename_dict = {
                            'osmid': 'ID', 
                            'zip': 'ZIP', 
                            'zone_type': 'Zone Type',
                            'demand_score': 'Demand Score',
                            'recommendation_status': 'Status',
                            'distance_to_nearest_charger': 'Distance to Nearest (m)',
                            'chargers_in_radius': 'Nearby Chargers',
                            'traffic': 'Traffic Volume'
                        }
                        
                        summary_df = summary_df.rename(columns={k: v for k, v in rename_dict.items() if k in summary_df.columns})
                        
                        # Apply conditional formatting to Status column if it exists
                        if 'Status' in summary_df.columns:
                            def status_color(val):
                                if val == 'Highly Recommended':
                                    return 'background-color: #ffcccc'  # Light red
                                elif val == 'Recommended':
                                    return 'background-color: #ffedcc'  # Light orange
                                else:
                                    return 'background-color: #cce6ff'  # Light blue
                            
                            # Display the summary table with conditional formatting
                            st.dataframe(summary_df.style
                                        .format(formatter={
                                            'Demand Score': '{:.3f}',
                                            'Distance to Nearest (m)': '{:.0f}',
                                            'Traffic Volume': '{:.0f}'
                                        })
                                        .applymap(status_color, subset=['Status']))
                        else:
                            # Display the summary table without conditional formatting
                            st.dataframe(summary_df.style.format(formatter={
                                'Demand Score': '{:.3f}',
                                'Distance to Nearest (m)': '{:.0f}',
                                'Traffic Volume': '{:.0f}'
                            }))
                    
                    with tab2:
                        # Display full data table with all columns
                        st.subheader("Complete Data for Recommended Locations")
                        
                        # Show all columns except geometry 
                        display_cols = [col for col in top_recommendations.columns if col != 'geometry']
                        
                        # Add rank column
                        full_df = top_recommendations[display_cols].copy()
                        full_df['rank'] = range(1, len(full_df) + 1)
                        full_df = full_df.set_index('rank')
                        
                        # Format numeric columns
                        st.dataframe(full_df.style.format(formatter={col: '{:.3f}' for col in full_df.select_dtypes('float').columns}))
                    
                    # Display visualization of factor contributions if ML analysis is used
                    if use_ml_analysis and 'recommendation_status' in top_recommendations.columns:
                        st.subheader("Factor Contribution Analysis")
                        
                        # Identify score columns
                        score_columns = [
                            'distance_score', 
                            'radius_score', 
                            'cs_total_score', 
                            'traffic_score', 
                            'population_score', 
                            'income_score', 
                            'zoning_score'
                        ]
                        
                        # Ensure all required columns exist
                        available_score_cols = [col for col in score_columns if col in top_recommendations.columns]
                        
                        if len(available_score_cols) > 0:
                            # Create columns for visualizations
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Create a heatmap of factor scores for top recommendations
                                st.subheader("Factor Score Heatmap")
                                
                                # Prepare data for heatmap
                                heatmap_data = top_recommendations[available_score_cols].head(10).copy()  # Limit to top 10
                                
                                # Better column labels for readability
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
                                
                                # Create a heatmap
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt='.2f', linewidths=.5, ax=ax)
                                plt.title('Factor Contribution Scores by Location')
                                plt.ylabel('Location Rank')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Add factor weight explanation
                                st.markdown("""
                                **Factor Weights:**
                                - **Distance** (0.5): Higher scores for locations farther from existing chargers
                                - **Nearby Chargers** (-0.3): Negative weight - fewer nearby chargers is better
                                - **Charging Points** (-0.2): Negative weight - fewer existing points is better
                                - **Traffic** (0.1): Higher scores for locations with more traffic
                                - **Population** (0.2): Higher scores for areas with greater population density
                                - **Income** (0.1): Higher scores for higher income areas
                                - **Zoning** (0.1): Higher scores for commercial and mixed-use zones
                                """)
                            
                            with col2:
                                # Create a radar chart for the top recommendation
                                st.subheader("Top Location Profile")
                                
                                # Use our safe radar chart function
                                plot_radar_in_streamlit(top_recommendations, available_score_cols)
                    
                    # Add download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Add download button for the data
                        csv = full_df.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Data as CSV",
                            data=csv,
                            file_name="ev_recommendations.csv",
                            mime="text/csv",
                        )
                    
                    with col2:
                        # Add download button for the map
                        if st.button("Download Map as HTML"):
                            map_html = m._repr_html_()
                            with open("ev_demand_map.html", "w") as f:
                                f.write(map_html)
                            st.success("Map saved as ev_demand_map.html")
                            
                            # Provide download link
                            with open("ev_demand_map.html", "rb") as f:
                                st.download_button(
                                    label="Click to Download Map",
                                    data=f,
                                    file_name="ev_demand_map.html",
                                    mime="text/html"
                                )
            
            else:
                # Show instructions when first loading the page
                st.info("Use the filters in the sidebar to select options, then click 'Show Recommendations' to generate the map.")
                
                # Show a sample explanation of the ML features
                st.markdown("""
                ### How the Machine Learning Analysis Works

                When you use the Machine Learning Analysis option, the app will:

                1. **Cluster Locations**: Group similar locations based on their characteristics using K-means clustering
                2. **Define Thresholds**: Automatically determine appropriate thresholds for each factor
                3. **Generate Natural Language**: Translate data points into human-readable explanations
                4. **Determine Recommendations**: Classify locations as "Highly Recommended," "Recommended," or "Not Recommended"
                5. **Create Visualizations**: Show factor contributions through heatmaps and radar charts

                The ML analysis helps identify patterns that might not be obvious from the raw scores alone, providing a more nuanced understanding of which locations truly deserve priority.
                """)
    
    except Exception as e:
        st.error(f"Error loading supporting data: {e}")
        st.info("Please make sure your data files exist and are properly formatted.")
        st.exception(e)  # Show detailed error for debugging.

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.info("Please make sure your data is properly loaded and contains the required columns.")
    st.exception(e)  # Show detailed error for debugging.
