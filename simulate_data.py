import numpy as np


def get_training_set(d, mu_0, mu_1, propensity, count):
    """
    Simulates a training set
    :param d: dimension of samples
    :param mu_0: response function for treated samples
    :param mu_1: response function for untreated samples
    :param propensity: propensity score
    :param count: number of samples to be generated
    :return: simulated data; with feature values, treatment assignment and outcome
    """
    # Features values sampled from N(0,1)
    x = list(map(lambda a: list(a), np.random.normal(loc=0, scale=1, size=(count, d))))

    # Treatment assigned based on the propensity
    w = list(map(lambda a: np.random.binomial(n=1, p=(propensity(a))), x))

    # Outcome based on treatment assignment and response functions, with added noise
    y = list(map(lambda a: (mu_0(a[0]) if a[1] == 0 else mu_1(a[0])) + np.random.normal(loc=0, scale=1), zip(x, w)))

    ret = {"X": x, "W": w, "Y": y}

    return ret


def get_test_set(d, mu_0, mu_1, propensity, count):
    """
    Simulates a test set
    :param d: dimension of samples
    :param mu_0: response function for treated samples
    :param mu_1: response function for untreated samples
    :param propensity: propensity score
    :param count: number of samples to be generated
    :return: simulated data; with feature values, treatment assignment, both counterfactual outcomes and the real one
    """
    # Features values sampled from N(0,1)
    x = list(map(lambda a: list(a), np.random.normal(loc=0, scale=1, size=(count, d))))

    # Treatment assignment based on the propensity
    w = list(map(lambda a: np.random.binomial(n=1, p=(propensity(a))), x))

    # Two counterfactual outcomes, with added noise
    y0 = list(map(lambda a: mu_0(a) + np.random.normal(loc=0, scale=1), x))
    y1 = list(map(lambda a: mu_1(a) + np.random.normal(loc=0, scale=1), x))

    # The outcome based on chosen treatment assignment
    y = list(map(lambda a: a[0] if a[2] == 0 else a[1], zip(y0, y1, w)))

    ret = {"X": x, "W": w, "Y": y, "Y0": y0, "Y1": y1}

    return ret
