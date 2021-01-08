import pandas as pd
import random as rnd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing as pp


# get the first data set as a data frame to cluster and a series which is used as the tag
def get_first_df():
    df = pd.read_csv("datasets/online_shoppers_intention.csv")
    # converting the string-type fields into int-type columns
    cols_to_convert = ['Month', 'Revenue', 'Weekend', 'VisitorType']
    convert_columns(df, cols_to_convert)

    # separate the data into to cluster and tag
    tag_fields = ['Revenue', 'Weekend', 'VisitorType']
    tag = df[tag_fields]
    to_cluster = df.drop(tag_fields, axis=1)
    return to_cluster, tag


# get the second data set as a data frame to cluster and a series which is used as the tag
def get_second_df():
    df = pd.read_csv("datasets/diabetic_data.csv")
    cols_to_convert = ['race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty', 'diag_1', 'diag_2',
                       'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
                       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                       'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                       'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                       'diabetesMed', 'readmitted']
    # converting the string-type fields into int-type columns
    convert_columns(df, cols_to_convert)

    # separate the data into to cluster and tag
    tag_fields = ['race', 'gender']
    tag = df[tag_fields]
    to_cluster = df.drop(tag_fields, axis=1)
    return to_cluster, tag


# get the third data set as a data frame to cluster and a series which is used as the tag
def get_third_df():
    df = pd.read_csv("datasets/e-shop clothing 2008.csv", delimiter=';')
    col_to_convert = ['page 2 (clothing model)']
    # converting the string-type fields into int-type columns
    convert_columns(df, col_to_convert)

    # separate the data into to cluster and tag
    tag_fields = ['country']
    tag = df[tag_fields]
    to_cluster = df.drop(tag_fields, axis=1)
    return to_cluster, tag


# convert the cols which are stored in 'cols_to_convert' from the data frame 'df' into columns such that each unique
# value in each column gets a unique int-type id
def PCA_alg(df):
    # get the data frame (the one we would like to cluster)
    df_values = df.values
    # before performing the PCA, normalize the values using standard scaler
    normalized_df_values = pp.StandardScaler().fit_transform(df_values)
    # create a PCA object, which reduces data into 2 dimensions
    pca = PCA(n_components=2)
    # do the PCA on the data and return the new reduced columns data frame with the columns names 'PC1' and 'PC2'
    pc = pca.fit_transform(normalized_df_values)
    return pd.DataFrame(data=pc, columns=['PC1', 'PC2'])


# perform a PCA for reducing the dimension of the data to 2
def convert_columns(df, cols_to_convert):
    for col_name in cols_to_convert:
        col = df[col_name].to_list()
        already_converted = {}
        curr_id = 0
        for i in range(len(col)):
            if col[i] not in already_converted.keys():
                already_converted[col[i]] = curr_id
                curr_id += 1
            col[i] = already_converted[col[i]]
        new_col = pd.Series(col, name=col_name)
        df.update(new_col)


# get a data frames for the clustering and for the tag according to the data set number (valid values - 1 to 3 -
# otherwise, throw an exception) if needed, the 'n_samples' parameter let you choose exact number of samples from the
# data set, randomly (its default value is None, means to return the whole data frame)
def get_data(dataset_num, n_samples=None):
    if dataset_num == 1:
        data = get_first_df()
    elif dataset_num == 2:
        data = get_second_df()
    elif dataset_num == 3:
        data = get_third_df()
    else:
        raise Exception("Dataset number is between 1 to 3")

    if n_samples is not None:
        samples_indices = rnd.sample(range(len(data[0])), n_samples)
        to_cluster, tag = data
        data = [to_cluster.loc[samples_indices], tag.loc[samples_indices]]
    data[0] = PCA_alg(data[0])
    return data


# get the data frame for the clustering (after PCA)
def get_df_to_cluster(data):
    return data[0]


# get the data frame which is used as a tag
def get_tag(data):
    return data[1]


# get the "real" labels of the data
def get_labels(tag):
    labels = []
    already_converted = {}
    curr_cluster_num = 0
    for r in tag.values:
        if str(r) not in already_converted.keys():
            already_converted[str(r)] = curr_cluster_num
            curr_cluster_num += 1
        labels.append(already_converted[str(r)])
    return labels


# get the "real" number of clusters for the data
def get_num_of_clusters(tag):
    return len(np.unique(get_labels(tag)))
