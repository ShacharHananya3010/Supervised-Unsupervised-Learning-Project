from Fuzzy_C_means import fcm_alg
from GMM import gmm_alg
from Hierarchical_clustering import agglomerative_clustering_alg
from KMeans import k_means_alg
from Spectral_clustering import spectral_clustering_alg
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd


# returns a dictionary that maps 1-5 to the different clustering algorithms' names
def ordered_clustering_alg():
    return {1: 'K-Means', 2: 'Fuzzy C Means', 3: 'GMM', 4: 'Agglomerative clustering', 5: 'Spectral clustering'}


# returns a list of 15 ami scores of the algorithm 'alg' on the data set which is number is 'dataset_num'
def get_ami_scores(dataset_num, alg):
    ami_scores = []
    for i in range(15):
        if alg == 'K-Means':
            ami_scores.append(k_means_alg(dataset_num, num_of_samples=6000, calc_ami_score=True))
        elif alg == 'Fuzzy C Means':
            ami_scores.append(fcm_alg(dataset_num, num_of_samples=6000, calc_ami_score=True))
        elif alg == 'GMM':
            ami_scores.append(gmm_alg(dataset_num, num_of_samples=6000, calc_ami_score=True))
        elif alg == 'Agglomerative clustering':
            ami_scores.append(agglomerative_clustering_alg(dataset_num, num_of_samples=6000, calc_ami_score=True))
        else:
            ami_scores.append(spectral_clustering_alg(dataset_num, num_of_samples=4000, calc_ami_score=True))

    return ami_scores


# t test that check if the mean of the first algorithm's ami (Adjusted Mutual Information) scores is greater than
# this of the second one. Return the p value of the test
def t_test(alg1_ami_scores, alg2_ami_scores):
    _, double_p = ttest_ind(alg1_ami_scores, alg2_ami_scores, equal_var=False)
    mean_ami_1 = np.mean(alg1_ami_scores)
    mean_ami_2 = np.mean(alg2_ami_scores)

    if mean_ami_2 > mean_ami_1:
        return double_p / 2.
    return 1.0 - double_p / 2.


# the method gets the data set num, and perform statistical test (with significance level of 5%), on the ami scores
# of the different algorithms to find the best clustering algorithm for the relevant data set and return its name.
# Simultaneously, the results of the statistical tests, including their p value are reported in a text file.
def find_best_alg(dataset_num):
    # initialize the text that will be copied to the report file
    text = ""
    text += "Data set " + str(dataset_num) + "\n\n"
    # store the algorithms dictionary
    alg_dict = ordered_clustering_alg()
    ami_scores_dict = {}
    text += 'Adjusted Mutual Information Scores:\n'

    # store the ami scores of the different algorithms in a dictionary
    for alg in alg_dict.values():
        ami_scores_dict[alg] = get_ami_scores(dataset_num, alg)
        # add the ami scores to the report file
        text += alg + " ~~~ ["
        for x in ami_scores_dict[alg]:
            text += '%.6f, ' % x
        text = text[:-2]
        text += ']\n'

    text += "\n\n~~~~~~~\nTest No. 1 - " + alg_dict[1] + " vs " + alg_dict[2] + '\n'

    # perform a statistical test on the first 2 algorithms, store the p value and add to the report file
    p_val = t_test(ami_scores_dict[alg_dict[1]], ami_scores_dict[alg_dict[2]])
    text += "p value: " + str(p_val) + '\n'
    if p_val > 0.05:
        best_alg_num = 1
    else:
        best_alg_num = 2
    text += 'Better algorithm: ' + alg_dict[best_alg_num] + '\n'

    # for each of the next algorithms, perform statistical tests between their ami scores to the best one so far
    for i in range(3, len(alg_dict.keys())+1):
        text += "\n~~~~~~~\nTest No. {} - ".format(i-1) + alg_dict[i] + " vs " + alg_dict[best_alg_num] + '\n'
        p_val = t_test(ami_scores_dict[alg_dict[i]], ami_scores_dict[alg_dict[best_alg_num]])
        text += 'p value: ' + str(p_val) + '\n'
        if p_val > 0.05:
            best_alg_num = i
        text += 'Better algorithm: ' + alg_dict[best_alg_num] + '\n'

    # add the best algorithm to the report file
    text += "\nThe best algorithm for data set No. {} is ".format(dataset_num) + alg_dict[best_alg_num] \
            + " with ami score " + str(np.mean(ami_scores_dict[alg_dict[best_alg_num]])) + "\n"

    # copy the text to the report file
    f = open("statistical tests/Statistical Tests - Data set {}.txt".format(dataset_num), 'w')
    f.write(text)
    f.close()
    # return the best algorithm's name
    return alg_dict[best_alg_num]


if __name__ == '__main__':
    for i in range(1, 4):
        find_best_alg(i)


