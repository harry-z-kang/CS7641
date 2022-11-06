import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np

# Code taken from scikit-learn examples (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
def silhouette_analysis(n_clusters, features, target, name, dataset_name):
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.5, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(features) + (n_clusters + 1) * 10])
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(features)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(features, cluster_labels)
    print(f"For n_clusters={n_clusters}, The average silhouette_score is: {silhouette_avg}")

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        features[:, 0], features[:, 1], marker=".",
        s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0], centers[:, 1], marker="o",
        c="white", alpha=1, s=200, edgecolor="k"
    )

    for i, c in enumerate(centers):
        ax2.scatter(
            c[0], c[1], marker=f"${i}$",
            alpha=1, s=50, edgecolor="k"
        )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        f"Silhouette analysis for KMeans clustering on sample data with the best No. of clusters = {n_clusters}",
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(f'images/{dataset_name}/kmeans_silhouette_analysis_{name}.png')
    plt.show()

    return clusterer

def train_neural_network(nn_classifier, features_train, target_train, features_test, target_test, name, dataset_name):
    param_grid = {
      'alpha': np.logspace(-3, 3, 7),
      'hidden_layer_sizes': np.arange(2, 25, 2)
    }
    nn_classifier_best = GridSearchCV(nn_classifier, param_grid=param_grid, cv=4)

    start_time = time.time()
    nn_classifier_best.fit(features_train, target_train)
    end_time = time.time()
    time_train = end_time-start_time
    print("Best params for neural network:",
          nn_classifier_best.best_params_)
    print("Time to train:", time_train)

    start_time = time.time()
    classifier_accuracy = accuracy_score(
        target_test, nn_classifier_best.predict(features_test))
    end_time = time.time()
    time_infer = end_time-start_time
    print("Accuracy for best neural network:", classifier_accuracy)
    print("Time to infer:", time_infer)

    nn_classifier_learning = MLPClassifier(
        random_state=42, max_iter=2000,
        hidden_layer_sizes=nn_classifier_best.best_params_['hidden_layer_sizes'],
        alpha=nn_classifier_best.best_params_['alpha']
    )
    _, train_scores, test_scores = learning_curve(
        nn_classifier_learning,
        features_train,target_train,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=4
    )

    plt.figure()
    plt.plot(np.linspace(0.1, 1.0, 10)*100,np.mean(train_scores, axis=1), label='Train Score')
    plt.plot(np.linspace(0.1,1.0,10)*100, np.mean(test_scores, axis=1), label='CV Score')
    plt.legend()
    plt.title("Learning Curve (Neural Network)")
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("Score")
    plt.xticks(np.linspace(0.1, 1.0, 10)*100)
    plt.grid()
    plt.savefig(f'images/{dataset_name}/neural_network_learning_curve_{name}.png')
    plt.show()
