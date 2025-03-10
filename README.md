# EV Charging Station Demand Analysis

A machine learning-powered application for identifying optimal locations for electric vehicle charging stations through geospatial analysis and multi-factor demand scoring.

![EV Charging Station Map](https://via.placeholder.com/800x400?text=EV+Charging+Station+Demand+Map)

## Overview

This Streamlit application helps urban planners, EV infrastructure developers, and policymakers identify the optimal locations for new electric vehicle charging stations. It integrates geospatial data, zoning information, demographic statistics, traffic patterns, and existing charging infrastructure to calculate a comprehensive demand score for potential charging station locations.

The application leverages machine learning (K-means clustering) to group similar locations and provide data-driven recommendations with detailed analysis of contributing factors.

## Features

- **Interactive Geospatial Visualization**: Multi-layered maps showing zoning, parking, and charging infrastructure
- **Machine Learning Analysis**: K-means clustering to identify location patterns and similarities
- **Multi-Factor Demand Scoring**: Comprehensive scoring system with weighted factors:
  - Distance to nearest charger (0.5)
  - Nearby chargers within radius (-0.3)
  - Existing charging points (-0.2)
  - Traffic volume (0.1)
  - Population density (0.2)
  - Income level (0.1)
  - Zoning suitability (0.1)
- **Customizable Filtering**: Filter recommendations by ZIP code and other parameters
- **Detailed Data Visualization**: Radar charts, heatmaps, and data tables for in-depth analysis
- **Natural Language Explanations**: AI-generated explanations of recommendations and insights
- **Data Export**: Download capabilities for recommendations and visualization

## Requirements

- Python 3.8+
- Streamlit 1.8+
- Folium 0.14+
- Pandas 1.3+
- GeoPandas 0.10+
- NumPy 1.20+
- Scikit-learn 1.0+
- Matplotlib 3.4+
- Seaborn 0.11+
- Branca 0.5+

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ev-charging-demand-analysis.git
   cd ev-charging-demand-analysis
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Prepare your data files (see Data Requirements section below)

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Data Requirements

The application requires the following data files to be placed in the application directory:

1. **features.csv**: Main dataset with location features and metrics
   - Required columns: `osmid`, `zip`, `geometry`, `zone_type`, `distance_to_nearest_charger`, `chargers_in_radius`, `cs_total`, `traffic`, `Population`, `Median Income`
   - The geometry should be in WKT format

2. **zoning_datasd.geojson**: Zoning information with polygons
   - Required attributes: `zone_name` or `zone_type`

3. **parking_data.geojson**: Public parking locations
   - Should contain point or polygon geometries

4. **lots.geojson**: Parking lots with zone information
   - Required attributes: `osmid`, `zone_type`

Example data format for `features.csv`:
```
osmid,zip,geometry,zone_type,distance_to_nearest_charger,...
123456,92101,POINT(-117.161 32.715),Commercial,450.2,...
```

## Usage Guide

### Overview Map Tab

The Overview Map provides a comprehensive view of the area's context:

1. **Zoning Areas**: Color-coded by zone type (commercial, residential, etc.)
2. **Public Parking Lots**: All available public parking
3. **Parking by Zone Type**: Parking lots with zoning information
4. **Distance to Nearest Charger**: Showing proximity to existing infrastructure

Use the layer control in the top-right corner to toggle different data layers.

### Recommendations Tab

The Recommendations tab provides data-driven charging station location suggestions:

1. **Filter Options**:
   - Select a specific ZIP code or view all areas
   - Adjust the number of recommendations to display
   - Toggle the Machine Learning Analysis option

2. **View the Results**:
   - Interactive map showing recommended locations
   - Color-coded markers based on recommendation status
   - Detailed analysis of each location's strengths and weaknesses
   - Factor contribution visualizations (radar charts, heatmaps)
   - Data tables with key metrics

3. **Export Options**:
   - Download the recommendations as CSV
   - Export the map as HTML for sharing

## Technical Implementation

### Demand Score Calculation

The application calculates a composite demand score based on multiple weighted factors:

```python
merged_features['demand_score'] = (
    merged_features['distance_score'] +
    merged_features['radius_score'] +
    merged_features['cs_total_score'] +
    merged_features['traffic_score'] +
    merged_features['population_score'] +
    merged_features['income_score'] +
    merged_features['zoning_score']
)
```

Factors and their weights:
- **Distance to nearest charger** (0.5): Higher scores for locations farther from existing chargers
- **Nearby chargers within radius** (-0.3): Negative weight - fewer nearby chargers is better
- **Existing charging points** (-0.2): Negative weight - fewer existing points is better
- **Traffic volume** (0.1): Higher scores for locations with more traffic
- **Population density** (0.2): Higher scores for areas with greater population density
- **Income level** (0.1): Higher scores for higher income areas
- **Zoning suitability** (0.1): Higher scores for commercial and mixed-use zones

### Machine Learning Analysis

The application uses K-means clustering to identify patterns in the data:

1. **Feature Selection**: Uses all score-related columns
2. **Standardization**: Normalizes features for equal contribution
3. **Dynamic Clustering**: Determines optimal number of clusters based on data size
4. **Cluster Analysis**: Identifies top factors for each cluster
5. **Recommendation Status**: Combines score thresholds and cluster information

```python
# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
```

### Visualization Components

The application creates multiple visualization types:

1. **Geospatial Maps**:
   - Folium-based interactive maps with multiple layers
   - Color-coded markers for recommendation status
   - Customizable view options

2. **Data Analysis Charts**:
   - Radar charts showing factor profiles for top locations
   - Heatmaps comparing factor contributions across locations
   - Data tables with conditional formatting

3. **Natural Language Explanations**:
   - Automatically generated HTML reports
   - Location-specific insights and recommendations
   - Machine learning pattern explanations

## Use Cases

This application supports various use cases:

1. **City Planning Departments**:
   - Identify optimal locations for public charging infrastructure
   - Support data-driven EV infrastructure expansion plans
   - Analyze coverage gaps in existing charging networks

2. **Charging Station Operators**:
   - Find profitable new station locations
   - Optimize network expansion strategy
   - Identify underserved high-potential areas

3. **Real Estate Developers**:
   - Assess potential for charging infrastructure in new developments
   - Evaluate property enhancement opportunities
   - Support EV-friendly development planning

4. **Utility Companies**:
   - Plan grid capacity upgrades for EV charging
   - Identify high-demand areas for infrastructure investment
   - Support strategic grid modernization efforts

## Customization

### Adding Custom Factors

To add new factors to the demand score calculation:

1. Add the new factor data to your `features.csv` file
2. Calculate percentile values for the new factor
3. Add a new weighted score component in the `load_data()` function:
   ```python
   merged_features['new_factor_score'] = merged_features['percentile_new_factor'] * weight_value
   ```
4. Add the new factor to the demand score calculation:
   ```python
   merged_features['demand_score'] = (
       # existing factors
       merged_features['new_factor_score'] +
   )
   ```

### Adjusting Factor Weights

To modify the importance of different factors:

1. Change the weight multipliers in the `load_data()` function:
   ```python
   merged_features['distance_score'] = merged_features['percentile_distance_to_nearest_charger'] * 0.6  # Increased from 0.5
   ```
2. Ensure that the weights reflect your priorities for charging station placement

## Future Improvements

Planned enhancements for future versions:

1. **Real-Time Data Integration**:
   - Live traffic data feeds
   - Current charging station usage statistics
   - Dynamic demand prediction

2. **Advanced Machine Learning**:
   - Predictive models for future EV adoption
   - Time-series analysis of charging patterns
   - Reinforcement learning for optimization

3. **Expanded Analysis Factors**:
   - Grid capacity and infrastructure costs
   - Environmental impact assessment
   - Renewable energy source proximity

4. **Enhanced User Interface**:
   - Mobile-friendly version for field assessment
   - User account system for saved analyses
   - Collaborative planning tools

## Troubleshooting

### Common Issues

1. **Map doesn't display properly**:
   - Ensure all geospatial data files exist and have valid geometries
   - Check that the coordinate reference system is set correctly (EPSG:4326)

2. **Missing data errors**:
   - Verify that all required columns exist in your data files
   - Ensure there are no null values in critical columns

3. **Performance issues with large datasets**:
   - Consider filtering data to a specific region
   - Increase memory allocation for the application
   - Optimize data loading with appropriate indices


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This application was inspired by the growing need for strategic EV charging infrastructure planning
- Special thanks to the open data communities providing valuable geospatial datasets
- Developed using the Streamlit framework for data science applications
