import numpy as np


class ExponentialWeights:
    """
    A log-space implementation of the exponential weights algorithm. After T
    rounds with losses in an interval of width H (i.e., in [a, a+H]), the regret
    of the algorithm can be bounded as

    Regret(T) <= stepsize * H^2 * T / 8 + ln(n_arms)

    Setting stepsize = sqrt(8 ln(N) / T) / H gives the best bound of

    Regret(T) <= 2 H sqrt(ln(N) T / 8)
    """

    def __init__(self, n_arms, stepsize):
        """
        Construct a new instance of the exponential weights algorithm with
        n_arms arms/actions/experts and the provided stepsize.
        """
        self.logweights = np.zeros(n_arms)
        self.stepsize = stepsize

    def update(self, losses):
        """
        Given a numpy vector of n_arms losses, updates the weights for each
        arm.
        """
        # Exponential weight updates become additive in log space
        self.logweights -= self.stepsize * losses

    def get_distribution(self):
        "Returns the distribution over arms given by the current weights."
        # We want to compute:
        #     w = np.exp(self.logweights)
        #     W = np.sum(w)
        #     p = w / W
        # In log space, we use the LogSumExp trick to compute log(W) avoiding
        # numerical overflow
        M = np.max(self.logweights)
        log_W = M + np.log(np.sum(np.exp(self.logweights - M)))
        # Now the log distribution vector is self.logweights - logW. We return
        # the exponential of that. Note this can't overflow since the values
        # are all in [0,1].
        return np.exp(self.logweights - log_W)
