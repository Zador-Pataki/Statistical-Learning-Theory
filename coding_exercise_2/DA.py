import sklearn as skl
from sklearn.utils.validation import check_is_fitted

import pandas as pd
import numpy as np
from treelib import Tree

import matplotlib.pyplot as plt

import time
import scipy


def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])

    return X, y


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, metric="euclidean"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T = None
        self.T_min = None

        self.cluster_centers = None
        self.cluster_probs = None

        self.n_eff_clusters = list()
        self.temperatures = list()
        self.distortions = list()


        self.bifurcation_tree = Tree()

        self.bifurcation_tree_cut_idx = 0
        self.bifurcation_tree_cut_idx_standby = None

        self.copy_cluster_centers = None
        self.T_split = 3.3

    ###### CUSTOM HELPER FUNCTIONS #####################################################################################
    def lambda_max(self, X):
        Cx = (1/X.shape[0])*X.T@X
        lambda_max_ = scipy.sparse.linalg.eigsh(Cx, which='LM', return_eigenvectors=False, k=1)[0]
        return lambda_max_

    def lambda_max_weighted(self, X, probs, y_i):
        for i in range(X.shape[0]):
            pxy = probs[i, y_i] / (X.shape[0] * self.cluster_probs[y_i])
            x = X[i,:][:,np.newaxis]
            if i == 0:
                Cxy= pxy * x@x.T
            else: Cxy += pxy * x@x.T

        lambda_max_ = scipy.sparse.linalg.eigsh(Cxy, which='LM', return_eigenvectors=False, k=1)[0]
        return lambda_max_
    ####################################################################################################################

    def fit(self, samples):
        idx = 1
        # TODO:
        alpha = 0.995


        # 1. SET LIMITS§
        Kmax = self.n_clusters
        coff_idx_fraction = 1/Kmax



        # 2. Initialize
        K=1
        self.cluster_centers = (np.sum(samples, axis=0)/samples.shape[0])[np.newaxis,:]
        T_crit = 2*self.lambda_max(samples-self.cluster_centers)
        self.T = 1.2*T_crit
        self.cluster_probs = [1]

        copy_cluster_centers = self.cluster_centers.copy()
        self.copy_cluster_centers = self.cluster_centers.copy()
        first_iter = True
        s=0

        self.T_min = T_crit/30


        while True:
            s+=1
            probs = self.predict(samples, self.T)

            ############################################################################################################
            if first_iter:
                self.distortions.append(np.linalg.norm(self.get_distance(samples,self.cluster_centers).min(axis=1)))
                self.n_eff_clusters.append(K)
                self.temperatures.append(self.T)
                
                self.bifurcation_tree.create_node("0", 'P0', data={'cluster_id': 0,'distance': [0.], 'parent_center': self.cluster_centers[0, :]})
            ############################################################################################################

            """if s%1000==0:
                try:print(self.T, len(self.cluster_probs), np.linalg.norm(copy_cluster_centers - self.cluster_centers), self.distortions[-1], len(self.temperatures))
                except:print(self.T, len(self.cluster_probs))"""



            if self.bifurcation_tree_cut_idx>0:
                self.bifurcation_tree_cut_idx = self.bifurcation_tree_cut_idx_standby + int((len(self.temperatures)-self.bifurcation_tree_cut_idx_standby) *coff_idx_fraction)


            copy_cluster_centers = self.cluster_centers.copy()

            first_iter = False




            for i in range(K): #
                # p(y_i)    :  dim(1)
                py_i = np.sum(probs[:, i])/samples.shape[0]
                y_i=samples * probs[:, i][:,np.newaxis]
                y_i = np.sum(y_i, axis=0)
                y_i = y_i[np.newaxis,:]
                self.cluster_centers[i, :] = y_i /(samples.shape[0] * py_i)



            # 4. Converge test
            if np.linalg.norm(copy_cluster_centers - self.cluster_centers)<1e-6:
                # 5
                ########################################################################################################
                self.distortions.append(np.linalg.norm(self.get_distance(samples,self.cluster_centers).min(axis=1)))
                self.n_eff_clusters.append(K)
                self.temperatures.append(self.T)

                for node in self.bifurcation_tree.all_nodes_itr():
                    if node.is_leaf():
                        if node.is_root():
                            node.data['distance'].append(node.data['distance'][-1] + np.linalg.norm(self.cluster_centers[0,:]-node.data['parent_center']))
                        else:
                            node_parent = self.bifurcation_tree[node.predecessor(self.bifurcation_tree.identifier)]
                            if node.data['plus']:
                                pm = 1
                            else:
                                pm = -1
                            node.data['distance'].append(node_parent.data['distance'][-1] + pm*np.linalg.norm(node_parent.data['parent_center']- self.cluster_centers[node.data['cluster_id']]))
                ########################################################################################################
                if self.temperatures[-1] <= self.T_min:
                    break

                else:
                    # 6. Cooling Step
                    self.T = alpha * self.T
                    if K < Kmax:
                        for i in range(K):
                            y_i = self.cluster_centers[i,:]
                            X_centered = samples - y_i[np.newaxis,:]
                            lambda_max_ = self.lambda_max_weighted(X_centered, probs, i)
                            if self.T < 2*lambda_max_:
                                #self.cluster_centers= np.vstack((self.cluster_centers, self.cluster_centers[i]+np.random.normal(scale=1, size=(self.cluster_centers[i].shape))))
                                self.cluster_centers= np.vstack((self.cluster_centers, self.cluster_centers[i]+np.ones(self.cluster_centers[i].shape)/10))
                                self.cluster_centers[i] = self.cluster_centers[i]
                                self.cluster_probs.append(self.cluster_probs[i]/2)
                                K+=1

                                ########################################################################################
                                for node in self.bifurcation_tree.all_nodes_itr():
                                    if node.data['cluster_id'] == i and node.is_leaf():
                                        parent_node = node

                                if idx>0:
                                    bool = True
                                else:
                                    bool = False
                                self.bifurcation_tree.create_node(parent_node.tag+'0', parent_node.identifier+'0', parent=parent_node.identifier,
                                                                  data={'cluster_id': parent_node.data['cluster_id'], 'distance': [], 'parent_center': self.cluster_centers[i, :], 'plus': bool})
                                self.bifurcation_tree.create_node(parent_node.tag+'1', parent_node.identifier+'1', parent=parent_node.identifier,
                                                                  data={'cluster_id': K-1,'distance': [], 'parent_center': self.cluster_centers[i, :], 'plus': not bool})
                                idx = -1*idx
                                ########################################################################################
                                if K==Kmax:

                                    self.bifurcation_tree_cut_idx = len(self.temperatures)
                                    self.bifurcation_tree_cut_idx_standby = len(self.temperatures)
                                #break

    def _calculate_cluster_probs(self, dist_mat, temperature):
        # TODO:
        probs = self.cluster_probs*np.exp(-dist_mat**2/temperature)
        probs = probs / probs.sum(axis=1)[:,np.newaxis]
        return probs

    def get_distance(self, samples, clusters):
        # TODO:
        D = np.zeros((samples.shape[0], clusters.shape[0]))
        for cluster_i in range(clusters.shape[0]):
            D[:, cluster_i] = np.linalg.norm(samples-clusters[cluster_i, :][np.newaxis, :], axis=1)
        return D

    def predict(self, samples, T=None):
        if T == None:
            T = self.T_min
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, T)
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        check_is_fitted(self, ["bifurcation_tree"])

        clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]
        for node in self.bifurcation_tree.all_nodes_itr():
            distances = node.data['distance']
            if node.is_leaf():
                node_x = node
                while True:
                    try:
                        node_x =  self.bifurcation_tree[node_x.predecessor(self.bifurcation_tree.identifier)]
                        if node.data['cluster_id'] == node_x.data['cluster_id']:
                            distances =  node_x.data['distance']+ distances
                    except:break
                clusters[node.data['cluster_id']] = distances

        max = 0
        for i in range(len(clusters)):
            if max < len(clusters[i]):
                max = len(clusters[i])
        for i in range(len(clusters)):
            diff = max - len(clusters[i])
            extension = []
            for j in range(diff):
                if not j == diff-1:extension.append(np.nan)
                else:
                    for node in self.bifurcation_tree.all_nodes_itr():
                        if node.data['cluster_id']==i:
                            node_x =  self.bifurcation_tree[node.predecessor(self.bifurcation_tree.identifier)]
                            break
                    extension.append(clusters[node_x.data['cluster_id']][j])

            clusters[i] = extension + clusters[i]


        cut_idx = self.bifurcation_tree_cut_idx

        beta = [1 / t for t in self.temperatures]
        plt.figure(figsize=(10, 5))
        for c_id, s in enumerate(clusters):
            plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
                     alpha=1, c='C%d' % int(c_id),
                     label='Cluster %d' % int(c_id))

        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()

    def plot_phase_diagram(self):
        t_max = np.log(max(self.temperatures))
        d_min = np.log(min(self.distortions))
        y_axis = [np.log(i) - d_min for i in self.distortions]
        x_axis = [t_max - np.log(i) for i in self.temperatures]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()

