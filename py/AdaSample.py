import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tqdm import tqdm


class AdaSample(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, clf):
        self.base_estimator_ = clone(clf)

    def fit(self, X, y, C=1, sampleFactor=1, seed=False, samp_frac=1, n_boost=25):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
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
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
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
