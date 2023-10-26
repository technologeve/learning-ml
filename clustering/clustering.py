""" Completing Microsoft Learn's Clustering Challenge."""

# External imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, MaxAbsScaler,
                                   Normalizer, QuantileTransformer)

# Dictionary for cluster colours
colour_dict = {0: "cornflowerblue", 1: "mediumvioletred",
               2: "goldenrod", 3: "blue"}


def perform_pca(scaled_features):
    """ Fit your features into two principle components. """

    pca = PCA(n_components=2).fit(scaled_features)
    features_2d = pca.transform(scaled_features)

    return features_2d


def calculate_wcss(scaled_features, max_clusters):
    """ Calculated the wcss score for a set of data points. """

    wcss = []

    for i in range(1, max_clusters+1):

        kmeans = KMeans(n_clusters=i)
        # Fit data
        kmeans.fit(scaled_features)
        # Note the wcss value
        wcss.append(kmeans.inertia_)

    return wcss


def plot_clusters(samples, clusters, no_of_clusters):
    """ Plot the clusters, each in different colours. """

    colors = [colour_dict[i] for i in clusters]

    for sample in range(len(clusters)):
        plt.scatter(samples[sample][0], samples[sample][1],
                    c=colors[sample])

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Assignments, {no_of_clusters} clusters')

    plt.show()


# Load dataset
data = pd.read_csv('data/clusters.csv')

# Normalize numeric features to range [0-1],
# trying a range of different scalers
not_scaled = data
std_scaled_features = StandardScaler().fit_transform(data)
min_max_scaled_features = MinMaxScaler().fit_transform(data)
max_abs_scaled_features = MaxAbsScaler().fit_transform(data)
norm_scaled_features = Normalizer().fit_transform(data)
quantile_scaled_features = QuantileTransformer().fit_transform(data)

# Perform principal component analysis to get two features
not_scaled_features = perform_pca(not_scaled)
std_features = perform_pca(std_scaled_features)
min_max_features = perform_pca(min_max_scaled_features)
max_abs_features = perform_pca(max_abs_scaled_features)
norm_features = perform_pca(norm_scaled_features)
quantile_features = perform_pca(quantile_scaled_features)

# Viualise these two components, with each of the scalers
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
plt.suptitle("Visualising our data after PCA", weight="bold")
fig.supxlabel("First component")
fig.supylabel("Second component")

ax[0, 0].scatter(not_scaled_features.T[0], not_scaled_features.T[1], s=1)
ax[0, 0].set_title("Not scaled")

ax[0, 1].scatter(std_features.T[0], std_features.T[1], s=1)
ax[0, 1].set_title("Standard scaler")

ax[0, 2].scatter(min_max_features.T[0], min_max_features.T[1], s=1)
ax[0, 2].set_title("Min Max scaler")

ax[1, 0].scatter(max_abs_features.T[0], max_abs_features.T[1], s=1)
ax[1, 0].set_title("Max abs scaler")

ax[1, 1].scatter(norm_features.T[0], norm_features.T[1], s=1)
ax[1, 1].set_title("Normalized")

ax[1, 2].scatter(quantile_features.T[0], quantile_features.T[1], s=1)
ax[1, 2].set_title("Quantile scaled")

plt.show()

# Create 10 models with 1 to 10 clusters
unscaled_wcss = calculate_wcss(not_scaled_features, 10)
std_wcss = calculate_wcss(std_features, 10)
min_max_wcss = calculate_wcss(min_max_features, 10)
max_abs_wcss = calculate_wcss(max_abs_features, 10)
norm_wcss = calculate_wcss(norm_features, 10)
quantile_wcss = calculate_wcss(quantile_features, 10)

# Plot the WCSS values onto a line graph
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
plt.suptitle('WCSS by Clusters', weight="bold")
fig.supxlabel('Number of clusters')
fig.supylabel('WCSS')

ax[0, 0].plot(range(1, 11), unscaled_wcss, lw=1)
ax[0, 0].set_title("Not scaled")

ax[0, 1].plot(range(1, 11), std_wcss, lw=1)
ax[0, 1].set_title("Standard scaler")

ax[0, 2].plot(range(1, 11), min_max_wcss, lw=1)
ax[0, 2].set_title("Min Max scaler")

ax[1, 0].plot(range(1, 11), max_abs_wcss, lw=1)
ax[1, 0].set_title("Max abs scaler")

ax[1, 1].plot(range(1, 11), norm_wcss, lw=1)
ax[1, 1].set_title("Normalized")

ax[1, 2].plot(range(1, 11), quantile_wcss, lw=1)
ax[1, 2].set_title("Quantile scaled")

plt.show()

# From these plots, it appears the min max scaler has the lowest wcss score
# So we'll use data scaled with that method from now
plt.title('WCSS by Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.plot(range(1, 11), min_max_wcss, lw=1)
plt.show()

# From this plot, it looks like either 3 or 4 clusters would be best
# Let's visualise both:
# Three clusters
model_3 = KMeans(n_clusters=3, init='k-means++', n_init=100, max_iter=1000)
result_3 = model_3.fit_predict(min_max_features)

plot_clusters(min_max_features, result_3, 3)

# Four clusters
model_4 = KMeans(n_clusters=4, init='k-means++', n_init=100, max_iter=1000)
result_4 = model_4.fit_predict(min_max_features)

plot_clusters(min_max_features, result_4, 4)
