import operator
from copy import copy

from hmmlearn import hmm
from scipy.special import softmax


class GHMM_classifier():
    def __init__(self, n_components=2, n_iter=10):
        self.models = {}
        self.n_components = n_components
        self.hmm_model = hmm.GaussianHMM(n_components=self.n_components, n_iter=n_iter)

    def fit(self, X, Y, verbose=False, n_iter=10):
        """
        X: input sequence [[[x1,x2,.., xn]...]]
        Y: output classes [1, 2, 1, ...]
        """
        print("Detect classes:", set(Y))
        print("Prepare datasets...")
        X_Y = {}
        X_lens = {}
        for c in set(Y):
            X_Y[c] = []
            X_lens[c] = []

        for x, y in zip(X, Y):
            X_Y[y].extend(x)
            X_lens[y].append(len(x))

        for c in set(Y):
            print("Fit HMM for", c, " class")
            hmm_model = copy(self.hmm_model)
            hmm_model.fit(X_Y[c], X_lens[c])
            self.models[c] = hmm_model

    def _predict_scores(self, X):

        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with log likehood per class
        """
        X_seq = []
        X_lens = []
        for x in X:
            X_seq.extend(x)
            X_lens.append(len(x))

        scores = {}
        for k, v in self.models.items():
            scores[k] = v.score(X)

        return scores

    def predict_proba(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with probabilities per class
        """
        pred = self._predict_scores(X)

        keys = list(pred.keys())
        scores = softmax(list(pred.values()))

        return dict(zip(keys, scores))

    def predict(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: predicted class label
        """
        pred = self.predict_proba(X)

        return max(pred.items(), key=operator.itemgetter(1))[0]