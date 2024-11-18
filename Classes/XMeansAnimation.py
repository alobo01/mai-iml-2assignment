import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib.colors as colors
import os
from pathlib import Path
from Classes.XMeans import XMeansAlgorithm

class XMeansVisualizer(XMeansAlgorithm):
    def __init__(self, k_min: int = 2, k_max: int = 10, distance_metric: str = 'euclidean',
                 max_iter: int = 100):
        """
        Initialize X-means visualizer.

        Args:
            k_min: Minimum number of clusters
            k_max: Maximum number of clusters
            distance_metric: Distance metric to use
            max_iter: Maximum number of iterations for each K-means run
            split_threshold: Threshold for cluster splitting decision
        """
        super().__init__(k_min, k_max, distance_metric, max_iter)
        self.animation_frames = []
        self.colors = list(colors.TABLEAU_COLORS.values())

    def record_frame(self, X: np.ndarray, labels: np.ndarray, title: str):
        """
        Record the current state for animation.

        Args:
            X: Input data
            labels: Current cluster assignments
            title: Frame title
        """
        self.animation_frames.append({
            'X': X.copy(),
            'labels': labels.copy(),
            'centroids': self.centroids.copy(),
            'k': self.k,
            'title': title,
            'frame_number': len(self.animation_frames)
        })

    def fit(self, X: np.ndarray) -> tuple:
        """
        Override fit method to record animation frames.
        """
        n_samples, n_features = X.shape
        if n_features != 2:
            raise ValueError("This visualizer only works with 2D data")

        # Clear previous animation frames
        self.animation_frames = []

        # Initialize with k_min random centroids
        random_indices = np.random.choice(n_samples, self.k_min, replace=False)
        self.centroids = X[random_indices].copy()
        self.k = self.k_min

        # Record initial state
        self.record_frame(X, np.zeros(n_samples), "Initial state")

        best_labels = None
        best_k = self.k_min
        best_bic = -np.inf

        iteration = 0
        while self.k <= self.k_max:
            # Run k-means with current configuration
            for i in range(self.max_iter):
                # Assign points to nearest centroids
                distances = self.distance(X, self.centroids)
                labels = np.argmin(distances, axis=1)

                # Record frame for point assignment
                self.record_frame(X, labels,
                                  f"K-means iteration {i + 1} (k={self.k})\nAssigning points to nearest centroids")

                # Update centroids
                new_centroids = np.zeros((self.k, n_features))
                for j in range(self.k):
                    cluster_points = X[labels == j]
                    if len(cluster_points) > 0:
                        new_centroids[j] = cluster_points.mean(axis=0)

                # Record frame for centroid update
                self.centroids = new_centroids
                self.record_frame(X, labels,
                                  f"K-means iteration {i + 1} (k={self.k})\nUpdating centroids")

                # Check for convergence
                if np.allclose(self.centroids, new_centroids):
                    break

            current_bic = self.compute_bic(X, labels, self.k)

            # Update best solution if necessary
            if current_bic > best_bic:
                best_bic = current_bic
                best_labels = labels.copy()
                best_k = self.k
                self.record_frame(X, labels,
                                  f"New best clustering found (k={self.k})\nBIC: {current_bic:.2f}")

            # Try splitting each cluster
            new_centroids = []
            split_occurred = False

            for j in range(self.k):
                cluster_points = X[labels == j]
                should_split, split_centroids = self.should_split_cluster(
                    X, cluster_points, self.centroids[j]
                )

                if should_split and self.k + 1 <= self.k_max:
                    new_centroids.extend(split_centroids)
                    split_occurred = True
                    # Record frame for cluster split
                    self.record_frame(X, labels,
                                      f"Splitting cluster {j + 1}\n(k={self.k}→{self.k + 1})")
                else:
                    new_centroids.append(self.centroids[j])

            if not split_occurred:
                break

            # Update centroids and k for next iteration
            self.centroids = np.array(new_centroids)
            self.k = len(self.centroids)
            iteration += 1

        # Record final frame
        self.record_frame(X, best_labels, f"Final clustering (k={best_k})\nBIC: {best_bic:.2f}")
        return best_labels, best_k

    def save_frames(self, output_dir: str = "frames", figsize=(12, 8), dpi=150):
        """
        Save individual frames as PNG files with consistent size.

        Args:
            output_dir: Directory to save frames
            figsize: Figure size (width, height)
            dpi: DPI for saved images
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Calculate figure size in pixels
        width_px = int(figsize[0] * dpi)
        height_px = int(figsize[1] * dpi)

        for frame in self.animation_frames:
            # Create new figure for each frame
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111)

            X = frame['X']
            labels = frame['labels']
            centroids = frame['centroids']
            k = frame['k']

            # Set plot limits with padding
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Plot data points
            for i in range(k):
                mask = labels == i
                ax.scatter(X[mask, 0], X[mask, 1], c=[self.colors[i % len(self.colors)]],
                           alpha=0.6, label=f'Cluster {i + 1}')

            # Plot centroids with larger markers
            ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=200,
                       linewidths=3, label='Centroids')

            # Add title and legend
            ax.set_title(frame['title'], pad=20, fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add axes labels
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')

            # Adjust layout
            plt.tight_layout()

            # Save frame with consistent size
            frame_number = str(frame['frame_number']).zfill(4)
            save_path = f"{output_dir}/frame_{frame_number}.png"

            # Save with specific dimensions
            fig.set_size_inches(figsize)
            fig.savefig(save_path, bbox_inches='tight', dpi=dpi)
            plt.close(fig)

    def create_gif(self, output_path: str = "xmeans_animation.gif", fps: int = 2):
        """
        Create GIF from saved frames using imageio.

        Args:
            output_path: Path to save the GIF
            fps: Frames per second for the GIF
        """
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir("frames") if f.endswith('.png')])

        if not frame_files:
            raise ValueError("No frames found in the frames directory")

        # Read first frame to get dimensions
        first_frame = imageio.imread(f"frames/{frame_files[0]}")
        target_size = first_frame.shape[:2]

        # Read and resize frames
        frames = []
        for frame_file in frame_files:
            frame = imageio.imread(f"frames/{frame_file}")
            # Ensure all frames have the same dimensions as the first frame
            if frame.shape[:2] != target_size:
                from PIL import Image
                frame_img = Image.fromarray(frame)
                frame_img = frame_img.resize(
                    (target_size[1], target_size[0]),
                    Image.Resampling.LANCZOS
                )
                frame = np.array(frame_img)
            frames.append(frame)

        # Save as GIF
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"GIF saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Generate sample 2D data
    np.random.seed(111)
    n_samples = 300

    # Create three clusters
    cluster1 = np.random.normal(loc=[-2, -2], scale=0.5, size=(n_samples//3, 2))
    cluster2 = np.random.normal(loc=[0, 2], scale=0.3, size=(n_samples//3, 2))
    cluster3 = np.random.normal(loc=[2, -1], scale=0.4, size=(n_samples//3, 2))
    X = np.vstack([cluster1, cluster2, cluster3])

    # Initialize and fit the visualizer
    visualizer = XMeansVisualizer(k_min=2, k_max=6)
    labels, k = visualizer.fit(X)

    # Save individual frames
    visualizer.save_frames(output_dir="frames", figsize=(12, 8), dpi=150)

    # Create slow GIF (2 FPS)
    visualizer.create_gif(output_path="xmeans_animation.gif", fps=2)
