"""Visualization utilities for PCA analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def plot_scree(pca: PCA, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a scree plot showing explained variance per component.
    
    Args:
        pca: Fitted PCA object
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # individual explained variance
    n_components = len(pca.explained_variance_ratio_)
    ax1.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot')
    ax1.grid(True, alpha=0.3)
    
    # cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, n_components + 1), cumsum, 'o-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_loadings(pca: PCA, feature_names: List[str], 
                  components: List[int] = [0, 1, 2],
                  figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot feature loadings for principal components.
    
    Args:
        pca: Fitted PCA object
        feature_names: List of feature names
        components: Which components to plot (indices)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_plots = len(components)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, comp_idx in enumerate(components):
        loadings = pca.components_[comp_idx]
        sorted_idx = np.argsort(np.abs(loadings))[::-1][:10]  # Top 10
        
        ax = axes[idx]
        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, loadings[sorted_idx])
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Loading')
        ax.set_title(f'PC{comp_idx + 1} Loadings\n({pca.explained_variance_ratio_[comp_idx]:.1%} variance)')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_pca_2d(df: pd.DataFrame, pc1: str = 'PC1', pc2: str = 'PC2',
                color_by: Optional[str] = None,
                figsize: Tuple[int, int] = (10, 8),
                title: str = 'PCA 2D Projection') -> plt.Figure:
    """
    Create a 2D scatter plot of PCA components.
    
    Args:
        df: DataFrame with PCA components and metadata
        pc1: Column name for first component
        pc2: Column name for second component
        color_by: Column to use for coloring points
        figsize: Figure size
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by and color_by in df.columns:
        if df[color_by].dtype in ['object', 'category']:
            categories = df[color_by].unique()
            for cat in categories:
                mask = df[color_by] == cat
                ax.scatter(df.loc[mask, pc1], df.loc[mask, pc2], 
                          label=cat, alpha=0.6, s=50)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            scatter = ax.scatter(df[pc1], df[pc2], c=df[color_by], 
                               cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        ax.scatter(df[pc1], df[pc2], alpha=0.6, s=50)
    
    ax.set_xlabel(pc1)
    ax.set_ylabel(pc2)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pca_3d_interactive(df: pd.DataFrame, 
                            pc1: str = 'PC1', pc2: str = 'PC2', pc3: str = 'PC3',
                            color_by: Optional[str] = None,
                            hover_data: Optional[List[str]] = None,
                            title: str = 'PCA 3D Projection'):
    """
    Create an interactive 3D scatter plot using plotly.
    
    Args:
        df: DataFrame with PCA components and metadata
        pc1, pc2, pc3: Column names for components
        color_by: Column to use for coloring points
        hover_data: Columns to show on hover
        title: Plot title
        
    Returns:
        Plotly figure
    """
    fig = px.scatter_3d(
        df, x=pc1, y=pc2, z=pc3,
        color=color_by if color_by else None,
        hover_data=hover_data,
        title=title,
        opacity=0.7
    )
    
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        scene=dict(
            xaxis_title=pc1,
            yaxis_title=pc2,
            zaxis_title=pc3
        ),
        height=700
    )
    
    return fig


def plot_biplot(df: pd.DataFrame, pca: PCA, feature_names: List[str],
                pc1_idx: int = 0, pc2_idx: int = 1,
                n_vectors: int = 10,
                figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create a biplot showing both instances and feature vectors.
    
    Args:
        df: DataFrame with PCA scores
        pca: Fitted PCA object
        feature_names: List of feature names
        pc1_idx, pc2_idx: Which components to plot
        n_vectors: Number of top feature vectors to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    pc1_col = f'PC{pc1_idx + 1}'
    pc2_col = f'PC{pc2_idx + 1}'
    
    ax.scatter(df[pc1_col], df[pc2_col], alpha=0.3, s=30, label='Instances')
    
    loadings1 = pca.components_[pc1_idx]
    loadings2 = pca.components_[pc2_idx]
    
    scale = max(df[pc1_col].max() - df[pc1_col].min(),
                df[pc2_col].max() - df[pc2_col].min()) * 0.3
    
    importance = np.sqrt(loadings1**2 + loadings2**2)
    top_idx = np.argsort(importance)[::-1][:n_vectors]
    
    for idx in top_idx:
        ax.arrow(0, 0, loadings1[idx] * scale, loadings2[idx] * scale,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
        ax.text(loadings1[idx] * scale * 1.1, loadings2[idx] * scale * 1.1,
               feature_names[idx], color='red', fontsize=9,
               ha='center', va='center')
    
    ax.set_xlabel(pc1_col)
    ax.set_ylabel(pc2_col)
    ax.set_title(f'Biplot: {pc1_col} vs {pc2_col}')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Create a correlation heatmap for features.
    
    Args:
        df: DataFrame with features
        features: List of feature columns to include (None = all numeric)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[features].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    return fig


def plot_cluster_comparison(df: pd.DataFrame, cluster_col: str,
                            features: List[str],
                            figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create box plots comparing feature distributions across clusters.
    
    Args:
        df: DataFrame with features and cluster labels
        cluster_col: Column containing cluster labels
        features: Features to compare
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        df.boxplot(column=feature, by=cluster_col, ax=ax)
        ax.set_title(f'{feature} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feature)
        plt.sca(ax)
        plt.xticks(rotation=45)
    
    # hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_pairplot(df: pd.DataFrame, components: List[str],
                  hue: Optional[str] = None) -> sns.PairGrid:
    """
    Create a pairplot of PCA components.

    Args:
        df: DataFrame with PCA components
        components: List of component columns to plot
        hue: Column to use for coloring

    Returns:
        Seaborn PairGrid
    """
    return sns.pairplot(df, vars=components, hue=hue,
                       diag_kind='kde', plot_kws={'alpha': 0.6})


def plot_feature_distributions(df: pd.DataFrame, features: List[str],
                                figsize: Tuple[int, int] = (15, 10),
                                bins: int = 30,
                                show_stats: bool = False) -> plt.Figure:
    """
    Create histograms for multiple features.

    Args:
        df: DataFrame with features
        features: List of feature columns to plot
        figsize: Figure size
        bins: Number of bins for histograms
        show_stats: Whether to show mean/std in titles

    Returns:
        Matplotlib figure
    """
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(features):
        if feature in df.columns:
            ax = axes[idx]
            df[feature].hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)

            if show_stats:
                mean_val = df[feature].mean()
                std_val = df[feature].std()
                ax.set_title(f'{feature}\n(mean={mean_val:.1f}, std={std_val:.1f})',
                           fontsize=10)
            else:
                ax.set_title(feature)

            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_repo_distribution(df: pd.DataFrame, repo_col: str = 'repo_type',
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Create a bar chart showing distribution of instances by repository.

    Args:
        df: DataFrame with repository column
        repo_col: Column name containing repository labels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    repo_counts = df[repo_col].value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    repo_counts.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Instances by Repository')
    ax.set_xlabel('Repository')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, repo_counts


def plot_repo_centroids(pca_df: pd.DataFrame, repo_col: str = 'repo_type',
                        pc1: str = 'PC1', pc2: str = 'PC2',
                        figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot PCA scatter with repository centroids highlighted.

    Args:
        pca_df: DataFrame with PCA scores and repository labels
        repo_col: Column containing repository labels
        pc1: Column name for first principal component
        pc2: Column name for second principal component
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Calculate centroids
    pc_cols = [pc1, pc2]
    if 'PC3' in pca_df.columns:
        pc_cols.append('PC3')

    repo_centroids = pca_df.groupby(repo_col)[pc_cols].mean()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all instances by repository
    for repo in pca_df[repo_col].unique():
        mask = pca_df[repo_col] == repo
        ax.scatter(pca_df.loc[mask, pc1], pca_df.loc[mask, pc2],
                  alpha=0.2, s=30, label=repo)

    # Plot centroids
    for repo, row in repo_centroids.iterrows():
        ax.scatter(row[pc1], row[pc2],
                  s=300, marker='*', edgecolors='black', linewidths=2,
                  label=f'{repo} (center)', zorder=10)

    ax.set_xlabel(pc1)
    ax.set_ylabel(pc2)
    ax.set_title('Repository Centroids in PCA Space')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig

