import numpy as np
import matplotlib.pyplot as plt
import generate_swiss_roll_dataset


def run(X, color):
    X_centered = center_dataset(X)

    n_samples = X.shape[0]
    covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_eigenvectors, sorted_eigenvalues = sort_eigenvectors(eigenvectors, eigenvalues)

    X_projected = apply_dot_product_of_eigenvectors_on_dataset(X_centered, color, sorted_eigenvectors)

    visualize_before_and_after(X, X_projected, color)

    return X_projected


def center_dataset(X: np.ndarray):
    column_means = np.mean(X, axis=0)
    X_centered = X - column_means
    return X_centered


def sort_eigenvectors(eigenvectors: np.ndarray, eigenvalues: np.ndarray):
    sorted_indices = np.argsort(eigenvalues)[::-1]  # sort ascending then reverse -> descending

    sorted_eigenvalues = eigenvectors[sorted_indices]

    sorted_eigenvectors = eigenvectors[:, sorted_indices]  # [:, x] means keep all rows but reorder the columns

    return sorted_eigenvectors, sorted_eigenvalues


def apply_dot_product_of_eigenvectors_on_dataset(X: np.ndarray, color: np.ndarray, eigenvectors: np.ndarray):
    # compress to 2D, so choose first 2 eigenvectors (PC1 and PC2)
    top_eigenvectors = eigenvectors[:, 0:2]

    # project X on 2D
    X_projected = np.dot(X, top_eigenvectors)

    plt.figure(figsize=(8, 6))

    # [:, 0] is the new X-axis (PC1) and [:, 1] is the new Y-axis (PC2)
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=color, cmap='viridis', s=20)

    return X_projected


def visualize_before_and_after(X_original: np.ndarray, X_after: np.ndarray, colors: np.ndarray):
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_original[:, 0], X_original[:, 1], X_original[:, 2], c=colors, cmap='viridis', s=20)
    ax1.set_title("Original 3D Swiss Roll")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(122)
    ax2.scatter(X_after[:, 0], X_after[:, 1], c=colors, cmap='viridis', s=20)
    ax2.set_title("PCA 2D Projection (Squished)")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    X, colors = generate_swiss_roll_dataset.generate()

    run(X, colors)
