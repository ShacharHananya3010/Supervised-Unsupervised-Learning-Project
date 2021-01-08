import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import Data_frames as d
from sklearn.metrics import adjusted_mutual_info_score


# perform agglomorative clustering for the data set which its number is stored in 'dataset_num', with 'num_of_clusters'
# clusters
def agglomerative_clustering_alg(dataset_num, num_of_samples=10000, num_of_clusters=None, get_figure=False,
                                 calc_ami_score=True):
    data = d.get_data(dataset_num, n_samples=num_of_samples)
    # store the data frame which is given by a dimension reduction (using PCA) into 2 dimensions of the original
    # data set
    df = d.get_df_to_cluster(data)
    tag = d.get_tag(data)

    # if the number of clusters isn't defined, choose it to be the "real" number of clusters (according to the tag)
    if num_of_clusters is None:
        num_of_clusters = d.get_num_of_clusters(tag)

    # create a AgglomeerativeClustering-type object with the relevant number of clusters and average linkage and fit
    # it to the data frame
    ac = AgglomerativeClustering(n_clusters=num_of_clusters, linkage='average').fit(df)
    # store the labels result after the fitting
    labels = ac.labels_

    if get_figure:
        # calculate the centroids of each cluster
        centroids = [[np.mean(df.values[labels == i][:, 0]), np.mean(df.values[labels == i][:, 1])] for i
                     in range(num_of_clusters)]

        # plot the clustered data and the centroid of each cluster
        plt.scatter(df['PC1'], df['PC2'], c=labels)
        plt.scatter([row[0] for row in centroids], [row[1] for row in centroids], c='black', marker='*',
                    label='centroid')
        title = 'DS{} - Agglomerative clustering'.format(dataset_num)
        fig_name = 'images/dataset {}/'.format(dataset_num) + title + " ({} clusters)".format(num_of_clusters)
        plt.title(title)
        plt.legend()

        # save the figure
        plt.savefig(fig_name)
        plt.show()

    # calculate the adjusted mutual info score of the clustering
    if calc_ami_score:
        labels_true = d.get_labels(tag)
        return adjusted_mutual_info_score(labels_true=labels_true, labels_pred=labels)