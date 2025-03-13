import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_weight_analysis_ui_1(merged_features, weight_optimizer):
    """
    Render the Weight Analysis UI component.
    
    Parameters:
    merged_features (DataFrame): The dataset with all features.
    weight_optimizer (WeightOptimizer): Instance of the weight optimizer class.
    """
    st.header("Feature Weight Optimization")
    
    # Get the feature columns
    feature_columns = [
        'distance_score',
        'radius_score',
        'cs_total_score',
        'traffic_score',
        'population_score',
        'income_score',
        'zoning_score'
    ]
    
    # Get the friendly names for display
    friendly_names = {
        'distance_score': 'Distance to Charger',
        'radius_score': 'Nearby Chargers',
        'cs_total_score': 'Charging Station Count',
        'traffic_score': 'Traffic Volume',
        'population_score': 'Population Density',
        'income_score': 'Median Income',
        'zoning_score': 'Zoning Suitability'
    }
    
    # Current weights
    current_weights = {
        'distance_score': 0.5,
        'radius_score': -0.3,
        'cs_total_score': -0.2,
        'traffic_score': 0.1,
        'population_score': 0.2,
        'income_score': 0.1,
        'zoning_score': 0.1
    }
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Weight Optimization", 
        "Weight Comparison", 
        "Weight Impact Analysis",
        "Sensitivity Analysis"
    ])
    
    with tab1:
        st.subheader("Automatic Weight Optimization")
        
        st.markdown("""
        This section uses unsupervised machine learning techniques to automatically optimize the feature weights 
        based on the inherent patterns in your data. Since there is no target column, we use techniques like 
        Principal Component Analysis (PCA), Factor Analysis, and Clustering to extract importance from data structure.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            methods = st.multiselect(
                "Select Optimization Methods",
                options=["PCA", "Factor Analysis", "Cluster Analysis", "Correlation"],
                default=["PCA", "Factor Analysis", "Cluster Analysis", "Correlation"]
            )
            
            # Map the method names to the keys expected by the optimizer
            method_mapping = {
                "PCA": "pca",
                "Factor Analysis": "factor_analysis",
                "Cluster Analysis": "cluster",
                "Correlation": "correlation"
            }
            
            methods_to_use = [method_mapping[m] for m in methods]
        
        with col2:
            pca_components = st.slider(
                "Number of PCA/Factor Components",
                min_value=1,
                max_value=min(5, len(feature_columns)),
                value=1,
                help="Number of principal components or factors to use for weight extraction"
            )
            
            use_absolute = st.checkbox(
                "Use Absolute Values",
                value=True,
                help="Whether to use absolute values of loadings (ignoring sign)"
            )
        
        # Button to run optimization
        if st.button("Run Weight Optimization", key="run_opt"):
            with st.spinner("Running weight optimization..."):
                try:
                    # Create WeightOptimizer
                    optimizer = weight_optimizer(
                        data=merged_features,
                        feature_columns=feature_columns,
                        current_weights=current_weights
                    )
                    
                    # Get optimized weights
                    optimized_weights = optimizer.optimize_weights(methods=methods_to_use)
                    
                    # Store in session state
                    st.session_state.optimized_weights = optimized_weights
                    
                    # Display the weights
                    st.subheader("Optimized Weights")
                    
                    # Create a DataFrame for easy comparison
                    weights_df = pd.DataFrame(optimized_weights)
                    
                    # Add current weights
                    weights_df['Current'] = pd.Series(current_weights)
                    
                    # Format the DataFrame
                    weights_df = weights_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
                    
                    # Use friendly names
                    weights_df.index = [friendly_names.get(idx, idx) for idx in weights_df.index]
                    
                    # Display the weights
                    st.dataframe(weights_df)
                    
                    # Create charts
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Weight Comparison Chart")
                        fig = optimizer.plot_weight_comparison(optimized_weights)
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Weight Heatmap")
                        fig = optimizer.plot_weight_heatmap(optimized_weights)
                        st.pyplot(fig)
                    
                    # Interactive Plotly chart
                    st.subheader("Interactive Weight Comparison")
                    fig = optimizer.create_interactive_weight_comparison(optimized_weights)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error during weight optimization: {str(e)}")
        else:
            st.info("Click 'Run Weight Optimization' to generate optimized weights.")
    
    with tab2:
        st.subheader("Feature Weight Comparison")
        
        st.markdown("""
        This section allows you to compare the different weight sets and how they would affect the 
        importance of each feature in determining optimal EV charging station locations.
        """)
        
        # Initialize weights with current values for manual adjustment
        if 'manual_weights' not in st.session_state:
            st.session_state.manual_weights = current_weights.copy()
        
        # Allow manual weight adjustment using sliders
        st.markdown("### Manual Weight Adjustment")
        st.markdown("Adjust the weights for each feature to see how it affects the feature importance.")
        
        # Create columns for sliders
        cols = st.columns(3)
        
        # Create sliders for each feature
        manual_weights = {}
        for i, (feature, weight) in enumerate(current_weights.items()):
            col_idx = i % 3
            with cols[col_idx]:
                friendly_name = friendly_names.get(feature, feature)
                manual_weights[feature] = st.slider(
                    f"{friendly_name}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(weight),
                    step=0.1,
                    key=f"manual_{feature}"
                )
        
        # Normalize manual weights
        sum_abs_weights = sum(abs(w) for w in manual_weights.values())
        if sum_abs_weights > 0:
            manual_weights = {k: v/sum_abs_weights for k, v in manual_weights.items()}
        
        # Store manual weights in session state
        st.session_state.manual_weights = manual_weights
        
        # If optimized weights exist, display comparison
        if 'optimized_weights' in st.session_state:
            # Create a comparison table
            comparison_data = st.session_state.optimized_weights.copy()
            comparison_data['Manual'] = manual_weights
            
            # Create a DataFrame for easy comparison
            weights_df = pd.DataFrame(comparison_data)
            
            # Add current weights
            weights_df['Current'] = pd.Series(current_weights)
            
            # Format the DataFrame
            weights_df = weights_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
            
            # Use friendly names
            weights_df.index = [friendly_names.get(idx, idx) for idx in weights_df.index]
            
            # Display the weights
            st.subheader("Weight Comparison Table")
            st.dataframe(weights_df)
            
            # Create a radar chart for weight comparison
            radar_fig = go.Figure()
            
            # Add traces for each weight set
            for method, weights in comparison_data.items():
                # Convert to list and close the loop by repeating first value
                weight_values = list(weights.values())
                weight_values.append(weight_values[0])
                
                # Use friendly names for display
                feature_names = [friendly_names.get(f, f) for f in weights.keys()]
                feature_names.append(feature_names[0])  # Close the loop
                
                # Add trace
                radar_fig.add_trace(go.Scatterpolar(
                    r=weight_values,
                    theta=feature_names,
                    fill='toself',
                    name=method
                ))
            
            # Update layout
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[-0.5, 0.5]
                    )),
                showlegend=True,
                title="Feature Weight Comparison (Radar Chart)",
                height=600
            )
            
            st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab3:
        st.subheader("Weight Impact Analysis")
        
        st.markdown("""
        This section analyzes how different weight sets impact the demand score and recommendations.
        """)
        
        # Need optimized weights to do impact analysis
        if 'optimized_weights' in st.session_state:
            try:
                # Get all weight sets including manual weights
                all_weights = st.session_state.optimized_weights.copy()
                all_weights['Manual'] = st.session_state.manual_weights
                all_weights['Current'] = current_weights
                
                # Create WeightOptimizer
                optimizer = weight_optimizer(
                    data=merged_features,
                    feature_columns=feature_columns,
                    current_weights=current_weights
                )
                
                # Calculate impact on demand scores
                demand_scores = optimizer.feature_impact_analysis(all_weights)
                
                # Extract original demand score if available
                if 'demand_score' in merged_features.columns:
                    demand_scores['Original'] = merged_features['demand_score']
                
                # Display basic statistics of the different scores
                st.subheader("Demand Score Statistics")
                
                score_stats = demand_scores.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                st.dataframe(score_stats)
                
                # Plot histograms of the demand scores
                st.subheader("Demand Score Distributions")
                
                # Prepare data for plotly
                fig = make_subplots(rows=1, cols=1)
                
                for col in demand_scores.columns:
                    fig.add_trace(
                        go.Histogram(
                            x=demand_scores[col],
                            name=col,
                            opacity=0.7,
                            nbinsx=30
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    barmode='overlay',
                    title="Distribution of Demand Scores by Weight Set",
                    xaxis_title="Demand Score",
                    yaxis_title="Count",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlations between scores
                correlation_matrix = demand_scores.corr()
                
                # Create a heatmap of correlations
                st.subheader("Correlation Between Different Weight Sets")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    correlation_matrix, 
                    annot=True, 
                    cmap='coolwarm',
                    vmin=-1, vmax=1,
                    ax=ax
                )
                plt.title('Correlation Between Demand Scores from Different Weight Sets')
                st.pyplot(fig)
                
                # Analyze top recommendations
                st.subheader("Impact on Top Recommendations")
                
                # Get the top 10 recommendations for each weight set
                top_recommendations = {}
                for column in demand_scores.columns:
                    # Sort by the demand score for this weight set
                    top_ids = demand_scores[column].sort_values(ascending=False).head(10).index
                    top_recommendations[column] = top_ids
                
                # Find common recommendations across weight sets
                all_top_ids = set()
                for ids in top_recommendations.values():
                    all_top_ids.update(ids)
                
                # Create a matrix showing which locations are in the top 10 for each weight set
                recommendation_matrix = pd.DataFrame(index=all_top_ids, columns=demand_scores.columns)
                
                for method, ids in top_recommendations.items():
                    recommendation_matrix[method] = recommendation_matrix.index.isin(ids).astype(int)
                
                # Calculate stability score for each location (how many methods recommend it)
                recommendation_matrix['stability_score'] = recommendation_matrix.sum(axis=1)
                recommendation_matrix = recommendation_matrix.sort_values('stability_score', ascending=False)
                
                # Get location details for the top IDs
                if 'osmid' in merged_features.columns:
                    location_details = merged_features.loc[recommendation_matrix.index, ['osmid', 'zip', 'zone_type']].copy()
                    recommendation_matrix = pd.concat([location_details, recommendation_matrix], axis=1)
                
                # Display the recommendation matrix
                st.dataframe(recommendation_matrix)
                
                # Visualize the stability of recommendations
                st.subheader("Recommendation Stability Analysis")
                
                # Create a heatmap of the recommendation matrix
                fig, ax = plt.subplots(figsize=(12, max(6, len(all_top_ids) * 0.4)))
                sns.heatmap(
                    recommendation_matrix.drop(['stability_score'] + list(location_details.columns), axis=1), 
                    cmap='Blues',
                    cbar_kws={'label': 'In Top 10'},
                    ax=ax
                )
                plt.title('Recommendation Stability Across Weight Sets')
                plt.ylabel('Location ID')
                plt.xlabel('Weight Set')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Create a bar chart of stability scores
                stability_data = recommendation_matrix[['osmid', 'stability_score']].sort_values('stability_score', ascending=False).head(15)
                
                fig = px.bar(
                    stability_data,
                    x='osmid',
                    y='stability_score',
                    title="Top 15 Most Stable Recommendations",
                    labels={'osmid': 'Location ID', 'stability_score': 'Stability Score'},
                    color='stability_score',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation of the analysis
                st.markdown("""
                ### Understanding the Analysis
                
                - **Demand Score Statistics:** Shows how different weight sets affect the distribution of scores.
                - **Demand Score Distributions:** Visualizes how different weights shift the overall distribution.
                - **Correlation Between Weight Sets:** Shows how similar the rankings are between different methods.
                - **Top Recommendations:** Identifies which locations are consistently recommended regardless of weights.
                - **Stability Score:** The number of weight sets that include a location in their top 10 (higher is more stable).
                
                Locations with high stability scores are robust recommendations that are likely to be good 
                candidates regardless of which exact weight values are used.
                """)
                
            except Exception as e:
                st.error(f"Error during impact analysis: {str(e)}")
        else:
            st.info("Please run weight optimization first in the 'Weight Optimization' tab.")
    
    with tab4:
        st.subheader("Sensitivity Analysis")
        
        st.markdown("""
        This tab allows you to analyze how sensitive the demand score is to changes in individual feature weights.
        By varying one weight at a time, you can see which features have the most impact on the final recommendations.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Select weight set to use as baseline
            weight_set_options = ["Current"]
            if 'optimized_weights' in st.session_state:
                weight_set_options.extend(list(st.session_state.optimized_weights.keys()))
                weight_set_options.append("Manual")
            
            selected_weight_set = st.selectbox(
                "Select Baseline Weight Set",
                options=weight_set_options,
                index=0
            )
            
            # Get the selected weights
            if selected_weight_set == "Current":
                base_weights = current_weights
            elif selected_weight_set == "Manual":
                base_weights = st.session_state.manual_weights if 'manual_weights' in st.session_state else current_weights
            else:
                base_weights = st.session_state.optimized_weights.get(selected_weight_set, current_weights)
        
        with col2:
            # Select feature to vary
            feature_options = list(base_weights.keys())
            feature_friendly_names = {k: friendly_names.get(k, k) for k in feature_options}
            
            feature_to_vary = st.selectbox(
                "Select Feature to Vary",
                options=feature_options,
                index=0,
                format_func=lambda x: feature_friendly_names[x]
            )
            
            # Select variation range
            variation_range = st.slider(
                "Variation Range",
                min_value=-0.5,
                max_value=0.5,
                value=(-0.3, 0.3),
                step=0.1,
                help="Range to vary the selected feature weight"
            )
        
        # Run sensitivity analysis on button click
        if st.button("Run Sensitivity Analysis", key="run_sens"):
            try:
                with st.spinner("Running sensitivity analysis..."):
                    # Create WeightOptimizer
                    optimizer = weight_optimizer(
                        data=merged_features,
                        feature_columns=feature_columns,
                        current_weights=current_weights
                    )
                    
                    # Run sensitivity analysis
                    sensitivity_results = optimizer.sensitivity_analysis(
                        base_weights=base_weights,
                        feature_to_vary=feature_to_vary,
                        variation_range=variation_range,
                        steps=20
                    )
                    
                    if sensitivity_results is not None:
                        # Plot the results
                        st.subheader(f"Sensitivity Analysis for {friendly_names.get(feature_to_vary, feature_to_vary)}")
                        
                        # Create the plot
                        fig = optimizer.plot_sensitivity_analysis(
                            sensitivity_results=sensitivity_results,
                            feature_name=friendly_names.get(feature_to_vary, feature_to_vary)
                        )
                        
                        st.pyplot(fig)
                        
                        # Display the results as a table
                        st.subheader("Sensitivity Analysis Data")
                        
                        # Format the table
                        display_results = sensitivity_results.copy()
                        display_results['delta'] = display_results['delta'].round(2)
                        display_results['adjusted_weight'] = display_results['adjusted_weight'].round(3)
                        display_results['avg_score'] = display_results['avg_score'].round(3)
                        display_results['min_score'] = display_results['min_score'].round(3)
                        display_results['max_score'] = display_results['max_score'].round(3)
                        
                        st.dataframe(display_results)
                        
                        # Analyze the impact on top recommendations
                        st.subheader("Impact on Top Recommendations")
                        
                        # For each weight variation, get the top 10 recommendations
                        all_top_ids = set()
                        top_recs_by_weight = {}
                        
                        for i, row in sensitivity_results.iterrows():
                            # Create a set of weights with this variation
                            test_weights = base_weights.copy()
                            adjusted_weight = row['adjusted_weight']
                            test_weights[feature_to_vary] = adjusted_weight
                            
                            # Normalize weights
                            total = sum(abs(w) for w in test_weights.values())
                            test_weights = {k: v/total for k, v in test_weights.items()}
                            
                            # Calculate score
                            score = np.zeros(len(merged_features))
                            for feat, weight in test_weights.items():
                                if feat in merged_features.columns:
                                    score += merged_features[feat].fillna(0) * weight
                            
                            # Get top 10 indices
                            top_indices = np.argsort(-score)[:10]
                            top_ids = merged_features.iloc[top_indices].index.tolist()
                            
                            # Store the result
                            variation_label = f"{feature_to_vary}={adjusted_weight:.2f}"
                            top_recs_by_weight[variation_label] = top_ids
                            all_top_ids.update(top_ids)
                        
                        # Create a matrix showing which locations are in the top 10 for each weight variation
                        rec_stability = pd.DataFrame(index=all_top_ids, columns=list(top_recs_by_weight.keys()))
                        
                        for variation, ids in top_recs_by_weight.items():
                            rec_stability[variation] = rec_stability.index.isin(ids).astype(int)
                        
                        # Calculate stability score
                        rec_stability['stability_score'] = rec_stability.sum(axis=1)
                        rec_stability = rec_stability.sort_values('stability_score', ascending=False)
                        
                        # Get location details for the top IDs
                        if 'osmid' in merged_features.columns:
                            location_details = merged_features.loc[rec_stability.index, ['osmid', 'zip', 'zone_type']].copy()
                            rec_stability = pd.concat([location_details, rec_stability], axis=1)
                        
                        # Display the recommendation stability matrix
                        st.dataframe(rec_stability)
                        
                        # Create a heatmap visualization
                        if len(all_top_ids) > 0:
                            # Ensure reasonable figure size
                            fig_height = min(20, max(8, len(all_top_ids) * 0.4))
                            
                            fig, ax = plt.subplots(figsize=(12, fig_height))
                            sns.heatmap(
                                rec_stability.drop(['stability_score'] + list(location_details.columns), axis=1),
                                cmap='Blues',
                                cbar_kws={'label': 'In Top 10'},
                                ax=ax
                            )
                            plt.title(f'Recommendation Stability Across {feature_to_vary} Weight Variations')
                            plt.ylabel('Location ID')
                            plt.xlabel('Weight Variation')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Add interpretation
                        st.markdown(f"""
                        ### Interpretation
                        
                        This analysis shows how changing the weight for **{friendly_names.get(feature_to_vary, feature_to_vary)}** affects:
                        
                        1. The overall demand score distribution (graph above)
                        2. Which specific locations appear in the top 10 recommendations (heatmap)
                        
                        **Key insights:**
                        - The average demand score {
                            "increases" if sensitivity_results['avg_score'].iloc[-1] > sensitivity_results['avg_score'].iloc[0] else 
                            "decreases" if sensitivity_results['avg_score'].iloc[-1] < sensitivity_results['avg_score'].iloc[0] else
                            "is relatively stable"
                        } as the weight increases
                        - Locations with high stability scores (near the top of the heatmap) are good candidates regardless of this weight's exact value
                        - {"The recommendations are highly sensitive to this weight" if rec_stability['stability_score'].max() < len(top_recs_by_weight) * 0.5 else
                           "The recommendations are moderately sensitive to this weight" if rec_stability['stability_score'].max() < len(top_recs_by_weight) * 0.8 else
                           "The recommendations are robust to changes in this weight"}
                        """)
                    else:
                        st.error(f"Feature {feature_to_vary} not found in the weights.")
            except Exception as e:
                st.error(f"Error during sensitivity analysis: {str(e)}")
                st.exception(e)
        else:
            st.info("Click 'Run Sensitivity Analysis' to analyze weight sensitivity.")