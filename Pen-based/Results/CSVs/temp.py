import pandas as pd
import plotly.express as px


def plot_3d_clusters_interactive(coordinates_file, clusters_file, cluster_column):
    # Load the coordinates
    coordinates = pd.read_csv(coordinates_file, usecols=["PC1", "PC2", "PC3"])

    # Load the clusters
    clusters = pd.read_csv(clusters_file, usecols=[cluster_column])
    clusters = clusters.rename(columns={cluster_column: "Cluster"})  # Rename column for clarity

    if len(coordinates) != len(clusters):
        raise ValueError("Mismatch in the number of records between coordinate and cluster files.")

    # Combine coordinates and cluster data
    data = pd.concat([coordinates, clusters], axis=1)

    # Create an interactive 3D scatter plot
    fig = px.scatter_3d(
        data,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Set3,
        title="Interactive 3D Cluster Visualization"
    )

    # Enhance layout
    fig.update_traces(marker=dict(size=1, opacity=0.8))
    fig.update_layout(scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3"
    ))

    # Show the interactive plot
    fig.show()


# Example usage:
# Replace 'coordinates.csv' and 'clusters.csv' with your actual filenames, and 'XMeans(..._0)' with the desired column
plot_3d_clusters_interactive("../../../Mushroom/Preprocessing/mushroom_pca.csv", "xmeans_cluster_labels.csv",
                 "XMeans(max_clusters=300, repeat_kmeans=1, use_kmeans_plus_plus=False)_0")
