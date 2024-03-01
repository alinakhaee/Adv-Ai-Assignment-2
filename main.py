from sklearn.metrics import silhouette_score
import matplotlib.markers as mmarkers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

datasets = ['circles0.3', 'moons1', 'spiral1', 'twogaussians42', 'halfkernel', 'threecircles']
clustering_algorithms = {
    'KMeans': KMeans(n_clusters=2),
    'EM': GaussianMixture(n_components=2),
    'Spectral': SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=10)
}
marker_styles = ["o", "s", "^"]
colormap = ["red", "blue", "orange"]


def load_dataset(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    coordinates = data[:, :-1]
    labels = data[:, -1]
    return coordinates, labels


def scatter_with_markers_list(x, y, ax=None, m=None, **kw):
    if not ax: ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def generate_threecircles():
    data1, labels1 = make_circles(n_samples=100,
                                  shuffle=True,
                                  noise=0.01,
                                  random_state=42)

    center = [[3, 4]]
    data2, labels2 = make_blobs(n_samples=100,
                                cluster_std=0.2,
                                centers=center,
                                random_state=1)

    for i in range(len(center) - 1, -1, -1):
        labels2[labels2 == 0 + i] = i + 2

    y_threecircles = np.concatenate([labels1, labels2])
    data1 = data1 * [1.2, 1.8] + [3, 4]
    X_threecircles = np.concatenate([data1, data2], axis=0)
    return X_threecircles, y_threecircles


def plot_clusters(X, y_true, y_pred, dataset_name, algorithm_name):
    clusters = np.concatenate((y_pred[:, None], y_true[:, None]), axis=1)
    markers = []
    colors = []
    for cluster_id, label in clusters:
        marker = marker_styles[int(cluster_id)]
        colors.append(colormap[int(label)])
        markers.append(marker)

    scatter_with_markers_list(X[:, 0], X[:, 1], c=colors, m=markers)
    plt.title(f"Clustering of {dataset_name} with {algorithm_name})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig('.\\graphs\\' + f"Clustering of {dataset_name} with {algorithm_name})" + '.png', bbox_inches='tight')
    plt.show()


def calculate_silhouette_scores(data, k_range):
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        scores.append(silhouette_score(data, kmeans.labels_))
    return scores


X_threecircles, y_threecircles = generate_threecircles()

for dataset in datasets:
    print(f'Dataset: {dataset}')
    if dataset == 'threecircles':
        coordinates = X_threecircles
        labels = y_threecircles
    else:
        file_path = f'..\\SampleDatasets\\{dataset}.csv'
        coordinates, labels = load_dataset(file_path)

    for algorithm_name, algorithm in clustering_algorithms.items():
        if dataset == 'threecircles':
            if algorithm_name == 'EM':
                algorithm.set_params(n_components=3)
            else:
                algorithm.set_params(n_clusters=3)
        algorithm.fit(coordinates)
        if algorithm_name == 'Spectral':
            y_pred = algorithm.fit_predict(coordinates)
        else:
            algorithm.fit(coordinates)
            y_pred = algorithm.predict(coordinates)

        plot_clusters(coordinates, labels, y_pred, dataset, algorithm_name)

for dataset in datasets:
    if dataset == 'threecircles':
        coordinates = X_threecircles
        labels = y_threecircles
    else:
        file_path = f'..\\SampleDatasets\\{dataset}.csv'
        coordinates, labels = load_dataset(file_path)
    k_range = range(2, 10)  # Range of k values to try
    scores = calculate_silhouette_scores(coordinates, k_range)
    best_k = np.argmax(scores) + 2
    print(f"\n## Best k for {dataset} using silhouette score: {best_k}")

    # Plot silhouette scores vs number of clusters (k)
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, scores, marker='o', label=f"{dataset} scores")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title(f"Silhouette scores for k-means on {dataset}")
    plt.legend()
    plt.grid(True)
    plt.savefig('.\\graphs\\' + f"Silhouette scores for k-means on {dataset}" + '.png', bbox_inches='tight')
    plt.show()
