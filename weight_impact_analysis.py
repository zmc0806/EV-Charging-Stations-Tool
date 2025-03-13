import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class WeightImpactAnalyzer:
    """
    Class for analyzing how changes in feature weights impact the demand score calculation
    and the resulting recommendations for EV charging station locations.
    """
    def __init__(self, data, feature_columns, current_weights):
        """
        Initialize the weight impact analyzer.
        
        Parameters:
        data (DataFrame): The dataset containing features.
        feature_columns (list): List of feature column names.
        current_weights (dict): Current feature weights.
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.current_weights = current_weights
        
        # Ensure all feature columns exist in the data
        self.available_features = [col for col in feature_columns if col in self.data.columns]
    
    def calculate_demand_score(self, weights):
        """
        Calculate demand score using specified weights.
        
        Parameters:
        weights (dict): Feature weights to use.
        
        Returns:
        Series: Calculated demand scores.
        """
        # Initialize scores as zeros
        scores = np.zeros(len(self.data))
        
        # Calculate weighted sum for valid features
        for feature, weight in weights.items():
            if feature in self.data.columns:
                scores += self.data[feature].fillna(0) * weight
        
        return pd.Series(scores, index=self.data.index)
    
    def compare_weight_sets(self, weight_sets):
        """
        Compare multiple sets of weights by calculating demand scores and analyzing differences.
        
        Parameters:
        weight_sets (dict): Dictionary of weight sets to compare, where each key is a name
                           and each value is a dictionary of weights.
        
        Returns:
        DataFrame: Data with demand scores calculated for each weight set.
        """
        # Calculate demand score for each weight set
        scores = pd.DataFrame(index=self.data.index)
        
        for name, weights in weight_sets.items():
            scores[f"{name}_score"] = self.calculate_demand_score(weights)
        
        return scores
    
    def rank_correlation_analysis(self, weight_sets):
        """
        Analyze the rank correlation between different weight sets.
        
        Parameters:
        weight_sets (dict): Dictionary of weight sets to compare.
        
        Returns:
        tuple: (rank_correlation_matrix, rank_agreement_score)
        """
        # Calculate scores for each weight set
        scores = self.compare_weight_sets(weight_sets)
        
        # Calculate rank for each location under each weight set
        ranks = pd.DataFrame(index=scores.index)
        
        for col in scores.columns:
            ranks[col.replace('_score', '_rank')] = scores[col].rank(ascending=False)
        
        # Calculate rank correlation matrix
        rank_correlation = ranks.corr(method='spearman')
        
        # Calculate rank agreement score (average of off-diagonal elements)
        n = len(rank_correlation)
        off_diag_sum = rank_correlation.sum().sum() - n  # Subtract diagonal elements
        rank_agreement = off_diag_sum / (n * (n - 1))  # Divide by number of off-diagonal elements
        
        return rank_correlation, rank_agreement
    
    def recommendation_stability_analysis(self, weight_sets, top_n=10):
        """
        Analyze how stable the top recommendations are across different weight sets.
        
        Parameters:
        weight_sets (dict): Dictionary of weight sets to compare.
        top_n (int): Number of top recommendations to consider.
        
        Returns:
        tuple: (recommendation_matrix, stability_scores)
        """
        # Calculate scores for each weight set
        scores = self.compare_weight_sets(weight_sets)
        
        # Get top N recommendations for each weight set
        top_recommendations = {}
        all_recommended_ids = set()
        
        for name, weights in weight_sets.items():
            score_col = f"{name}_score"
            top_indices = scores[score_col].nlargest(top_n).index
            top_recommendations[name] = top_indices
            all_recommended_ids.update(top_indices)
        
        # Create a matrix showing which locations are in the top N for each weight set
        recommendation_matrix = pd.DataFrame(index=all_recommended_ids, columns=weight_sets.keys())
        
        for method, indices in top_recommendations.items():
            recommendation_matrix[method] = recommendation_matrix.index.isin(indices).astype(int)
        
        # Calculate stability score for each location (% of weight sets that recommend it)
        stability_scores = recommendation_matrix.sum(axis=1) / len(weight_sets)
        
        # Sort by stability score
        recommendation_matrix['stability_score'] = stability_scores
        recommendation_matrix = recommendation_matrix.sort_values('stability_score', ascending=False)
        
        return recommendation_matrix, stability_scores
    
    def weight_sensitivity_analysis(self, base_weights, feature, variation_range=(-0.5, 0.5), steps=10):
        """
        Analyze sensitivity of demand scores to variations in a single feature weight.
        
        Parameters:
        base_weights (dict): Base weights to modify.
        feature (str): Feature whose weight will be varied.
        variation_range (tuple): Range of variation (min_delta, max_delta).
        steps (int): Number of variation steps to analyze.
        
        Returns:
        DataFrame: Results of sensitivity analysis.
        """
        if feature not in base_weights or feature not in self.data.columns:
            return None
        
        # Generate weight variations
        variations = np.linspace(variation_range[0], variation_range[1], steps)
        
        results = []
        base_value = base_weights[feature]
        
        # Calculate scores for each variation
        for delta in variations:
            # Create modified weights
            test_weights = base_weights.copy()
            new_value = max(0, base_value + delta)  # Ensure non-negative
            test_weights[feature] = new_value
            
            # Normalize weights to sum to 1
            total = sum(abs(w) for w in test_weights.values())
            normalized_weights = {k: v/total for k, v in test_weights.items()}
            
            # Calculate demand score
            scores = self.calculate_demand_score(normalized_weights)
            
            # Calculate statistics
            results.append({
                'delta': delta,
                'adjusted_weight': new_value,
                'normalized_weight': normalized_weights[feature],
                'mean_score': scores.mean(),
                'median_score': scores.median(),
                'min_score': scores.min(),
                'max_score': scores.max(),
                'std_score': scores.std()
            })
        
        return pd.DataFrame(results)
    
    def recommendation_change_analysis(self, weight_sets, base_set, top_n=10):
        """
        Analyze how recommendations change from a base weight set to others.
        
        Parameters:
        weight_sets (dict): Dictionary of weight sets to compare.
        base_set (str): Name of the base weight set to compare against.
        top_n (int): Number of top recommendations to consider.
        
        Returns:
        dict: Analysis of recommendation changes.
        """
        if base_set not in weight_sets:
            return None
        
        # Calculate scores for each weight set
        scores = self.compare_weight_sets(weight_sets)
        
        # Get top N recommendations for each weight set
        top_recommendations = {}
        for name, weights in weight_sets.items():
            score_col = f"{name}_score"
            top_indices = scores[score_col].nlargest(top_n).index
            top_recommendations[name] = set(top_indices)
        
        # Calculate changes relative to base set
        base_recommendations = top_recommendations[base_set]
        changes = {}
        
        for name, recommendations in top_recommendations.items():
            if name != base_set:
                retained = recommendations.intersection(base_recommendations)
                added = recommendations - base_recommendations
                removed = base_recommendations - recommendations
                
                changes[name] = {
                    'retained': len(retained),
                    'added': len(added),
                    'removed': len(removed),
                    'retained_pct': len(retained) / top_n * 100,
                    'added_pct': len(added) / top_n * 100,
                    'removed_pct': len(removed) / top_n * 100,
                    'retained_ids': list(retained),
                    'added_ids': list(added),
                    'removed_ids': list(removed)
                }
        
        return changes
    
    def feature_contribution_analysis(self, weights):
        """
        Analyze how much each feature contributes to the final demand score.
        
        Parameters:
        weights (dict): Feature weights to analyze.
        
        Returns:
        DataFrame: Feature contributions to the demand score.
        """
        # Calculate contribution of each feature
        contributions = pd.DataFrame(index=self.data.index)
        
        for feature, weight in weights.items():
            if feature in self.data.columns:
                contributions[feature] = self.data[feature].fillna(0) * weight
        
        # Calculate total score
        contributions['total_score'] = contributions.sum(axis=1)
        
        # Calculate percentage contribution
        for feature in weights.keys():
            if feature in self.data.columns:
                # Handle division by zero
                contributions[f"{feature}_pct"] = np.where(
                    contributions['total_score'] != 0,
                    contributions[feature] / contributions['total_score'].abs() * 100,
                    0
                )
        
        return contributions
    
    def plot_weight_sensitivity_curve(self, sensitivity_results, feature_name):
        """
        Plot sensitivity analysis results.
        
        Parameters:
        sensitivity_results (DataFrame): Results from weight_sensitivity_analysis().
        feature_name (str): Name of the feature that was varied.
        
        Returns:
        matplotlib.figure.Figure: Figure with sensitivity curve.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot mean score with min/max range
        ax.plot(
            sensitivity_results['normalized_weight'],
            sensitivity_results['mean_score'],
            'o-',
            linewidth=2,
            label='Mean Score'
        )
        
        ax.fill_between(
            sensitivity_results['normalized_weight'],
            sensitivity_results['mean_score'] - sensitivity_results['std_score'],
            sensitivity_results['mean_score'] + sensitivity_results['std_score'],
            alpha=0.3,
            label='±1 Std Dev'
        )
        
        ax.fill_between(
            sensitivity_results['normalized_weight'],
            sensitivity_results['min_score'],
            sensitivity_results['max_score'],
            alpha=0.1,
            label='Min-Max Range'
        )
        
        # Add title and labels
        ax.set_title(f'Sensitivity Analysis for {feature_name} Weight')
        ax.set_xlabel('Normalized Weight')
        ax.set_ylabel('Demand Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    def plot_recommendation_changes(self, change_analysis, base_set):
        """
        Plot changes in recommendations compared to a base set.
        
        Parameters:
        change_analysis (dict): Output from recommendation_change_analysis().
        base_set (str): Name of the base weight set.
        
        Returns:
        matplotlib.figure.Figure: Figure showing recommendation changes.
        """
        # Extract data for plotting
        methods = list(change_analysis.keys())
        retained = [change_analysis[m]['retained_pct'] for m in methods]
        added = [change_analysis[m]['added_pct'] for m in methods]
        removed = [change_analysis[m]['removed_pct'] for m in methods]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot stacked bars
        ax.bar(methods, retained, label='Retained', color='green', alpha=0.7)
        ax.bar(methods, added, bottom=retained, label='Added (New)', color='blue', alpha=0.7)
        ax.bar(methods, removed, label='Removed', color='red', alpha=0.7)
        
        # Add labels and title
        ax.set_title(f'Recommendation Changes Compared to {base_set}')
        ax.set_xlabel('Weight Set')
        ax.set_ylabel('Percentage of Recommendations')
        ax.legend()
        ax.set_ylim(0, 100)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add percentage labels
        for i, method in enumerate(methods):
            ax.text(i, retained[i]/2, f"{retained[i]:.0f}%", ha='center', va='center', fontweight='bold')
            ax.text(i, retained[i] + added[i]/2, f"{added[i]:.0f}%", ha='center', va='center', fontweight='bold')
            ax.text(i, 100 - removed[i]/2, f"{removed[i]:.0f}%", ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_contributions(self, contributions, top_n=10):
        """
        Plot feature contributions to the demand score.
        
        Parameters:
        contributions (DataFrame): Output from feature_contribution_analysis().
        top_n (int): Number of top locations to analyze.
        
        Returns:
        matplotlib.figure.Figure: Figure showing feature contributions.
        """
        # Get top locations by total score
        top_locations = contributions.nlargest(top_n, 'total_score')
        
        # Extract percentage columns
        pct_columns = [col for col in top_locations.columns if col.endswith('_pct')]
        pct_data = top_locations[pct_columns].copy()
        
        # Rename columns to remove _pct suffix
        pct_data.columns = [col.replace('_pct', '') for col in pct_data.columns]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot stacked bars
        pct_data.T.plot(kind='bar', stacked=True, ax=ax)
        
        # Add labels and title
        ax.set_title('Feature Contribution to Demand Score (Top Locations)')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Contribution Percentage')
        ax.legend(title='Location ID')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_sensitivity_chart(self, sensitivity_results, feature_name):
        """
        Create an interactive plotly chart for sensitivity analysis.
        
        Parameters:
        sensitivity_results (DataFrame): Results from weight_sensitivity_analysis().
        feature_name (str): Name of the feature that was varied.
        
        Returns:
        plotly.graph_objects.Figure: Interactive sensitivity chart.
        """
        # Create figure
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['normalized_weight'],
                y=sensitivity_results['mean_score'],
                mode='lines+markers',
                name='Mean Score',
                line=dict(color='royalblue', width=3),
                marker=dict(size=8)
            )
        )
        
        # Add standard deviation range
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['normalized_weight'],
                y=sensitivity_results['mean_score'] + sensitivity_results['std_score'],
                mode='lines',
                name='+1 Std Dev',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['normalized_weight'],
                y=sensitivity_results['mean_score'] - sensitivity_results['std_score'],
                mode='lines',
                name='-1 Std Dev',
                fill='tonexty',
                fillcolor='rgba(68, 114, 196, 0.3)',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        # Add min-max range
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['normalized_weight'],
                y=sensitivity_results['max_score'],
                mode='lines',
                name='Max Score',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=sensitivity_results['normalized_weight'],
                y=sensitivity_results['min_score'],
                mode='lines',
                name='Min Score',
                fill='tonexty',
                fillcolor='rgba(68, 114, 196, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        # Add annotations for the base value
        mid_index = len(sensitivity_results) // 2
        base_weight = sensitivity_results.iloc[mid_index]['normalized_weight']
        
        fig.add_shape(
            type="line",
            x0=base_weight,
            y0=min(sensitivity_results['min_score']),
            x1=base_weight,
            y1=max(sensitivity_results['max_score']),
            line=dict(color="red", width=2, dash="dash")
        )
        
        fig.add_annotation(
            x=base_weight,
            y=max(sensitivity_results['max_score']),
            text="Base Weight",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
        
        # Update layout
        fig.update_layout(
            title=f'Sensitivity Analysis for {feature_name} Weight',
            xaxis_title='Normalized Weight',
            yaxis_title='Demand Score',
            height=600,
            width=900,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode="x unified"
        )
        
        # Add hover template
        fig.update_traces(
            hovertemplate="<b>Weight:</b> %{x:.3f}<br><b>Score:</b> %{y:.3f}<extra></extra>"
        )
        
        return fig
    
    def create_interactive_contribution_chart(self, contributions, top_n=10):
        """
        Create an interactive plotly chart for feature contributions.
        
        Parameters:
        contributions (DataFrame): Output from feature_contribution_analysis().
        top_n (int): Number of top locations to analyze.
        
        Returns:
        plotly.graph_objects.Figure: Interactive contribution chart.
        """
        # Get top locations by total score
        top_locations = contributions.nlargest(top_n, 'total_score')
        
        # Extract percentage columns
        pct_columns = [col for col in top_locations.columns if col.endswith('_pct')]
        
        # Prepare data for plotting
        plot_data = []
        
        for idx, row in top_locations.iterrows():
            location_id = idx
            
            for col in pct_columns:
                feature = col.replace('_pct', '')
                
                # Skip very small contributions for clarity
                if abs(row[col]) < 1:
                    continue
                
                plot_data.append({
                    'Location ID': location_id,
                    'Feature': feature,
                    'Contribution (%)': row[col],
                    'Absolute Contribution': abs(row[col]),
                    'Direction': 'Positive' if row[col] >= 0 else 'Negative'
                })
        
        # Convert to DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Create interactive bar chart
        fig = px.bar(
            plot_df,
            x='Location ID',
            y='Contribution (%)',
            color='Feature',
            title='Feature Contribution to Demand Score (Top Locations)',
            labels={'Location ID': 'Location ID', 'Contribution (%)': 'Contribution (%)'},
            height=600,
            barmode='relative'
        )
        
        # Update layout
        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            legend_title_text='Feature',
            hovermode="closest"
        )
        
        return fig
    
    def create_interactive_recommendation_change_chart(self, change_analysis, base_set):
        """
        Create an interactive plotly chart for recommendation changes.
        
        Parameters:
        change_analysis (dict): Output from recommendation_change_analysis().
        base_set (str): Name of the base weight set.
        
        Returns:
        plotly.graph_objects.Figure: Interactive recommendation change chart.
        """
        # Extract data for plotting
        methods = list(change_analysis.keys())
        retained = [change_analysis[m]['retained_pct'] for m in methods]
        added = [change_analysis[m]['added_pct'] for m in methods]
        removed = [change_analysis[m]['removed_pct'] for m in methods]
        
        # Create figure
        fig = go.Figure()
        
        # Add retained bars
        fig.add_trace(
            go.Bar(
                x=methods,
                y=retained,
                name='Retained',
                marker_color='green',
                text=[f"{x:.1f}%" for x in retained],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Retained: %{y:.1f}%<extra></extra>"
            )
        )
        
        # Add added bars
        fig.add_trace(
            go.Bar(
                x=methods,
                y=added,
                name='Added (New)',
                marker_color='blue',
                text=[f"{x:.1f}%" for x in added],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Added: %{y:.1f}%<extra></extra>"
            )
        )
        
        # Add removed bars
        fig.add_trace(
            go.Bar(
                x=methods,
                y=removed,
                name='Removed',
                marker_color='red',
                text=[f"{x:.1f}%" for x in removed],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Removed: %{y:.1f}%<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f'Recommendation Changes Compared to {base_set}',
            xaxis_title='Weight Set',
            yaxis_title='Percentage of Recommendations',
            barmode='group',
            height=500,
            yaxis=dict(
                range=[0, 100]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig