import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm


class AdaSample(BaseEstimator):
    """
    Basic scikit learn estimator wrapper for using AdaSampling
    Based on the R package AdaSampling and the conference paper:
    AdaSampling for positive-unlabeled and label noise learning with bioinformatics applications.
    Yang, P., Ormerod, J., Liu, W., Ma, C., Zomaya, A., Yang, J.(2018) [doi:10.1109/TCYB.2018.2816984]

    Parameters
    ----------
    clf - base classifier to perform AdaSampling on. Should be a scikit-learn compatible classifier
    with the methods - fit & predict_proba
    """

    def __init__(self, clf):
        self.base_estimator_ = clone(clf)

    def fit(self, X, y, C=1, sampleFactor=1, seed=False, n_rounds=25):
        """
        Fitting function
        Adheres to arguments of the original R package.

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values - 1 for positive, 0 for negative.
        C : number of ensemble classifiers (default: 1)
        sampleFactor : float, subsampling factor (default: 1)
        seed : bool, Whether to set seed
               if True, seed is set to iteration index (default: False)
        n_rounds: int, number of AdaSampling rounds (default: 25)

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.is_fitted_ = True

        # initialize sampling probablity
        Ps = np.where(y == 1)[0]
        Ns = np.where(y == 0)[0]

        pos_probs = np.ones_like(Ps) / Ps.shape[0]
        una_probas = np.ones_like(Ns) / Ns.shape[0]

        self.adaSamples_ = []
        print("Training AdaSamples..")
        for i in tqdm(range(n_boost)):  # TODO: Maybe till convergence?
            boost_inds_P = np.random.randint(0, Ps.shape[0],
                                             int(Ps.shape[0] * samp_frac))
            boost_inds_N = np.random.randint(0, Ns.shape[0],
                                             int(Ns.shape[0] * samp_frac))

            Ps_i = np.copy(Ps)[boost_inds_P]
            pos_probs_i = np.copy(pos_probs)[boost_inds_P]
            pos_probs_i = pos_probs_i / pos_probs_i.sum()
            Ns_i = np.copy(Ns)[boost_inds_N]
            una_probas_i = np.copy(una_probas)[boost_inds_N]
            una_probas_i = una_probas_i / una_probas_i.sum()

            self.adaSamples_.append(
                self._fit_single(X, y,
                                 Ps_i, Ns_i,
                                 pos_probs_i, una_probas_i,
                                 sampleFactor, seed=(i if seed else False))
            )
            probas = self.adaSamples_[-1].predict_proba(X)
            pos_probs = probas[Ps, 1] / probas[Ps, 1].sum()
            una_probas = probas[Ns, 0] / probas[Ns, 0].sum()

        print("Training {} Classifiers".format(C))
        self.estimators_ = []
        for i in tqdm(range(C)):
            self.estimators_.append(
                self._fit_single(X, y,
                                 Ps, Ns,
                                 pos_probs, una_probas,
                                 sampleFactor, seed=(i if seed else False))
            )
        return self

    def _fit_single(self, X, y, Ps, Ns, pos_probs, una_probs, sampleFactor, seed):
        """
        Helper function for fitting.
        Similar to singleIter.R

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values - 1 for positive, 0 for negative.
        Ps : array-like, shape (n_positives,) indeces of positives
        Ns : array-like, shape (n_negatives,) indeces of negatives
        pos_probs : array-like, shape (n_positives,)
                    probability of positives to belong to the positive class
        una_probs : array-like, shape (n_negatives,)
                    probability of unlabeled to belong to the negative class
        sampleFactor : float, subsampling factor (default: 1)
        seed : bool, Whether to set seed
               if True, seed is set to iteration index (default: False)

        Returns
        -------
        clf : fitted classifier object of type self.base_estimator_
        """
        clf = clone(self.base_estimator_)
        if type(seed) == int:
            np.random.seed(seed)

        sampleN = max(Ps.shape, Ns.shape)
        ids_p = np.random.choice(Ps, size=sampleFactor * sampleN, p=pos_probs)
        X_p = X[ids_p, :]
        y_p = y[ids_p]

        ids_n = np.random.choice(Ns, size=sampleFactor * sampleN, p=una_probs)
        X_n = X[ids_n, :]
        y_n = y[ids_n]

        X_train = np.vstack((X_p, X_n))
        y_train = np.concatenate((y_p, y_n))

        clf.fit(X_train, y_train)

        return clf

    def predict_proba(self, X, single=False):
        """
        Predicting function. predicts probailities for the ensemble.
        Can be used in AdaSingle mode by setting single=True.
        Then the function predicts based on the first classifier only.

        When single = False, returns the average of predictions for the ensemble.

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The training input samples.
        single : bool, whether to predict in mode AdaSingle (default: False)

        Returns
        -------
        y : ndarray, shape (n_samples,2)
            Returns probabilities to belong to the negative class (col 0)
            or the positive class (col 1)
            Similar to other scikit-learn classifiers.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')

        probas = np.zeros((X.shape[0], 2))
        if single:
            return self.estimators_[0].predict_proba(X)

        for clf in self.estimators_:
            probas += clf.predict_proba(X)

        probas = probas / len(self.estimators_)
        return probas
