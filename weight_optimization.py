import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class WeightOptimizer_1:
    """
    Class for optimizing feature weights using unsupervised learning techniques
    such as PCA and Factor Analysis, without requiring a target column.
    """
    def __init__(self, data, feature_columns, current_weights=None):
        """
        Initialize the weight optimizer.
        
        Parameters:
        data (DataFrame): The input data containing features.
        feature_columns (list): List of feature column names to be used for weight optimization.
        current_weights (dict): Current weights being used for each feature.
        """
        self.data = data.copy()
        self.feature_columns = [col for col in feature_columns if col in self.data.columns]
        
        # Initialize with current weights if provided
        self.current_weights = current_weights if current_weights else {}
        
        # Prepare the data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare the data for analysis by handling missing values and scaling."""
        # Extract feature data
        self.feature_data = self.data[self.feature_columns].copy()
        
        # Fill missing values with the median (more robust than mean)
        for col in self.feature_columns:
            self.feature_data[col] = self.feature_data[col].fillna(self.feature_data[col].median())
        
        # Scale the data
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.feature_data)
    
    def run_pca(self, n_components=None):
        """
        Run Principal Component Analysis on the feature data.
        
        Parameters:
        n_components (int): Number of principal components to use.
        
        Returns:
        tuple: (pca_model, explained_variance, component_weights)
        """
        # If n_components not specified, use all features
        if n_components is None:
            n_components = len(self.feature_columns)
        
        # Run PCA
        pca = PCA(n_components=n_components)
        pca.fit(self.scaled_data)
        
        # Get the explained variance
        explained_variance = pca.explained_variance_ratio_
        
        # Get the component weights
        component_weights = pd.DataFrame(
            pca.components_, 
            columns=self.feature_columns,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return pca, explained_variance, component_weights
    
    def run_factor_analysis(self, n_factors=None):
        """
        Run Factor Analysis on the feature data.
        
        Parameters:
        n_factors (int): Number of factors to use.
        
        Returns:
        tuple: (fa_model, component_weights)
        """
        # If n_factors not specified, use all features
        if n_factors is None:
            n_factors = len(self.feature_columns)
        
        # Run Factor Analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(self.scaled_data)
        
        # Get the component weights
        component_weights = pd.DataFrame(
            fa.components_, 
            columns=self.feature_columns,
            index=[f'Factor{i+1}' for i in range(n_factors)]
        )
        
        return fa, component_weights
    
    def derive_optimal_weights(self, method='pca', n_components=1, absolute=True):
        """
        Derive optimal weights from PCA or Factor Analysis.
        
        Parameters:
        method (str): Method to use, either 'pca' or 'factor_analysis'.
        n_components (int): Number of components/factors to use.
        absolute (bool): Whether to use absolute values of loadings.
        
        Returns:
        dict: Optimized weights for each feature.
        """
        if method == 'pca':
            # Run PCA and get the first principal component
            pca, _, component_weights = self.run_pca(n_components=n_components)
            loadings = component_weights.iloc[0]  # Use first component
        else:
            # Run Factor Analysis and get the first factor
            _, component_weights = self.run_factor_analysis(n_factors=n_components)
            loadings = component_weights.iloc[0]  # Use first factor
        
        # Use absolute values if requested
        if absolute:
            loadings = loadings.abs()
        
        # Normalize to sum to 1
        normalized_loadings = loadings / loadings.sum()
        
        # Convert to dictionary
        optimized_weights = normalized_loadings.to_dict()
        
        return optimized_weights
    
    def cluster_based_weights(self, n_clusters=3):
        """
        Use K-means clustering to identify feature importance based on cluster centroids.
        
        Parameters:
        n_clusters (int): Number of clusters to create.
        
        Returns:
        dict: Feature weights derived from cluster analysis.
        """
        # Run K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.scaled_data)
        
        # Get the centroids
        centroids = kmeans.cluster_centers_
        
        # Calculate the dispersion of each feature across centroids
        feature_dispersion = np.std(centroids, axis=0)
        
        # Normalize to sum to 1
        normalized_dispersion = feature_dispersion / np.sum(feature_dispersion)
        
        # Convert to dictionary
        cluster_weights = {feature: weight for feature, weight in zip(self.feature_columns, normalized_dispersion)}
        
        return cluster_weights
    
    def correlation_based_weights(self):
        """
        Derive weights based on correlation with overall score or other features.
        
        Returns:
        dict: Feature weights based on correlation strength.
        """
        correlation_matrix = self.feature_data.corr().abs()
        
        # For each feature, calculate the average correlation with all other features
        avg_correlations = correlation_matrix.mean()
        
        # Normalize to sum to 1
        normalized_correlations = avg_correlations / avg_correlations.sum()
        
        # Convert to dictionary
        correlation_weights = normalized_correlations.to_dict()
        
        return correlation_weights
    
    def optimize_weights(self, methods=None):
        """
        Optimize weights using multiple methods and return a consensus.
        
        Parameters:
        methods (list): List of methods to use, can include 'pca', 'factor_analysis', 'cluster', 'correlation'.
        
        Returns:
        dict: Optimized weights from all methods.
        """
        if methods is None:
            methods = ['pca', 'factor_analysis', 'cluster', 'correlation']
        
        all_weights = {}
        
        if 'pca' in methods:
            all_weights['PCA'] = self.derive_optimal_weights(method='pca')
        
        if 'factor_analysis' in methods:
            all_weights['Factor Analysis'] = self.derive_optimal_weights(method='factor_analysis')
        
        if 'cluster' in methods:
            all_weights['Cluster Analysis'] = self.cluster_based_weights()
        
        if 'correlation' in methods:
            all_weights['Correlation'] = self.correlation_based_weights()
        
        # Calculate consensus weights
        consensus_weights = {}
        for feature in self.feature_columns:
            feature_weights = [weights[feature] for weights in all_weights.values()]
            consensus_weights[feature] = np.mean(feature_weights)
        
        # Normalize consensus weights to sum to 1
        total = sum(consensus_weights.values())
        consensus_weights = {k: v/total for k, v in consensus_weights.items()}
        
        all_weights['Consensus'] = consensus_weights
        
        return all_weights
    
    def plot_weight_comparison(self, optimized_weights):
        """
        Create a plot comparing different weight optimization methods.
        
        Parameters:
        optimized_weights (dict): Dictionary of weight methods.
        
        Returns:
        matplotlib.figure.Figure: Figure object with the comparison plot.
        """
        # Convert to DataFrame for easier plotting
        weights_df = pd.DataFrame(optimized_weights)
        
        # Add current weights if available
        if self.current_weights:
            # Adjust to use only the features we have
            current_weights_subset = {k: v for k, v in self.current_weights.items() if k in self.feature_columns}
            weights_df['Current'] = pd.Series(current_weights_subset)
        
        # Transpose for better plotting
        weights_df = weights_df.T
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        weights_df.plot(kind='bar', ax=ax)
        plt.title('Feature Weight Comparison Across Methods')
        plt.xlabel('Method')
        plt.ylabel('Normalized Weight')
        plt.xticks(rotation=45)
        plt.legend(title='Feature')
        plt.tight_layout()
        
        return fig
    
    def plot_weight_heatmap(self, optimized_weights):
        """
        Create a heatmap of optimized weights.
        
        Parameters:
        optimized_weights (dict): Dictionary of weight methods.
        
        Returns:
        matplotlib.figure.Figure: Figure object with the heatmap.
        """
        # Convert to DataFrame for easier plotting
        weights_df = pd.DataFrame(optimized_weights)
        
        # Add current weights if available
        if self.current_weights:
            # Adjust to use only the features we have
            current_weights_subset = {k: v for k, v in self.current_weights.items() if k in self.feature_columns}
            weights_df['Current'] = pd.Series(current_weights_subset)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(weights_df, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
        plt.title('Feature Weight Heatmap Across Methods')
        plt.ylabel('Feature')
        plt.xlabel('Method')
        plt.tight_layout()
        
        return fig
    
    def create_interactive_weight_comparison(self, optimized_weights):
        """
        Create an interactive plotly visualization comparing weights.
        
        Parameters:
        optimized_weights (dict): Dictionary of weight methods.
        
        Returns:
        plotly.graph_objects.Figure: Plotly figure with interactive comparison.
        """
        # Convert to DataFrame for easier plotting
        weights_df = pd.DataFrame(optimized_weights)
        
        # Add current weights if available
        if self.current_weights:
            # Adjust to use only the features we have
            current_weights_subset = {k: v for k, v in self.current_weights.items() if k in self.feature_columns}
            weights_df['Current'] = pd.Series(current_weights_subset)
        
        # Create a figure with two subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Feature Weight Comparison by Method', 'Feature Weight Comparison by Feature'),
            vertical_spacing=0.2,
            specs=[[{"type": "bar"}], [{"type": "bar"}]]
        )
        
        # Add traces for each feature
        for feature in weights_df.index:
            fig.add_trace(
                go.Bar(
                    x=weights_df.columns, 
                    y=weights_df.loc[feature], 
                    name=feature,
                    text=weights_df.loc[feature].round(2),
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # Add traces for each method
        for method in weights_df.columns:
            fig.add_trace(
                go.Bar(
                    x=weights_df.index, 
                    y=weights_df[method], 
                    name=method,
                    text=weights_df[method].round(2),
                    textposition='auto'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800, 
            width=1000,
            barmode='group',
            legend_title_text='Legend',
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text='Method', row=1, col=1)
        fig.update_yaxes(title_text='Weight', row=1, col=1)
        
        fig.update_xaxes(title_text='Feature', row=2, col=1)
        fig.update_yaxes(title_text='Weight', row=2, col=1)
        
        return fig
    
    def feature_impact_analysis(self, weight_sets=None):
        """
        Analyze how different weight values impact the final demand score.
        
        Parameters:
        weight_sets (dict): Dictionary of weight sets to compare.
        
        Returns:
        DataFrame: Data with demand scores calculated using different weight sets.
        """
        if weight_sets is None:
            weight_sets = self.optimize_weights()
        
        # Initialize a DataFrame to store calculated demand scores
        demand_scores = pd.DataFrame(index=self.data.index)
        
        # Calculate demand score for each weight set
        for method_name, weights in weight_sets.items():
            # Calculate weighted sum for this weight set
            # Use only the features that have weights assigned
            score = np.zeros(len(self.data))
            for feature, weight in weights.items():
                if feature in self.data.columns:
                    score += self.data[feature].fillna(0) * weight
            
            demand_scores[f'{method_name}_score'] = score
        
        return demand_scores
    
    def sensitivity_analysis(self, base_weights, feature_to_vary, variation_range=(-0.5, 0.5), steps=10):
        """
        Perform sensitivity analysis by varying the weight of one feature.
        
        Parameters:
        base_weights (dict): Base set of weights to modify.
        feature_to_vary (str): Feature whose weight will be varied.
        variation_range (tuple): Range of variation (min_delta, max_delta).
        steps (int): Number of steps to evaluate.
        
        Returns:
        DataFrame: Results of the sensitivity analysis.
        """
        if feature_to_vary not in base_weights:
            return None
        
        # Create variation steps
        variations = np.linspace(variation_range[0], variation_range[1], steps)
        
        results = []
        base_value = base_weights[feature_to_vary]
        
        # For each variation
        for delta in variations:
            # Adjust the weight for the target feature
            test_weights = base_weights.copy()
            new_value = max(0, base_value + delta)  # Prevent negative weights
            test_weights[feature_to_vary] = new_value
            
            # Normalize weights to sum to 1
            total = sum(test_weights.values())
            test_weights = {k: v/total for k, v in test_weights.items()}
            
            # Calculate demand score
            score = np.zeros(len(self.data))
            for feature, weight in test_weights.items():
                if feature in self.data.columns:
                    score += self.data[feature].fillna(0) * weight
            
            # Calculate stats on the score
            avg_score = score.mean()
            min_score = score.min()
            max_score = score.max()
            
            # Store results
            results.append({
                'delta': delta,
                'adjusted_weight': test_weights[feature_to_vary],
                'avg_score': avg_score,
                'min_score': min_score,
                'max_score': max_score
            })
        
        return pd.DataFrame(results)
    
    def plot_sensitivity_analysis(self, sensitivity_results, feature_name):
        """
        Plot the results of sensitivity analysis.
        
        Parameters:
        sensitivity_results (DataFrame): Results from sensitivity_analysis().
        feature_name (str): Name of the feature that was varied.
        
        Returns:
        matplotlib.figure.Figure: Figure object with the sensitivity plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot average, min, and max scores
        ax.plot(sensitivity_results['adjusted_weight'], sensitivity_results['avg_score'], 'o-', label='Average Score')
        ax.fill_between(
            sensitivity_results['adjusted_weight'],
            sensitivity_results['min_score'],
            sensitivity_results['max_score'],
            alpha=0.2,
            label='Score Range'
        )
        
        ax.set_title(f'Sensitivity Analysis for {feature_name} Weight')
        ax.set_xlabel('Feature Weight')
        ax.set_ylabel('Demand Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig

# Function to get user-friendly feature names for display
def get_friendly_feature_names(feature_columns):
    """Convert feature column names to more readable names for display."""
    friendly_names = {
        'distance_score': 'Distance to Charger',
        'radius_score': 'Nearby Chargers',
        'cs_total_score': 'Charging Station Count',
        'traffic_score': 'Traffic Volume',
        'population_score': 'Population Density',
        'income_score': 'Median Income',
        'zoning_score': 'Zoning Suitability'
    }
    
    return {col: friendly_names.get(col, col.replace('_', ' ').title()) for col in feature_columns}