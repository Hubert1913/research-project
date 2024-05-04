import numpy as np


def get_training_set(d, mu_0, mu_1, propensity, count):
    ret = {"X": [], "W": [], "Y": []}
    for i in range(count):
        x = np.random.normal(loc=0, scale=1, size=d)

        w = np.random.binomial(n=1, p=(propensity(x)))

        ret["X"].append(x)
        ret["W"].append(w)
        if w == 0:
            ret["Y"].append(mu_0(x))
        else:
            ret["Y"].append(mu_1(x))

    return ret


def get_test_set(d, mu_0, mu_1, propensity, count):
    ret = {"X": [], "W": [], "Y": [], "Y0": [], "Y1": []}
    for i in range(count):
        x = np.random.normal(loc=0, scale=1, size=d)

        w = np.random.binomial(n=1, p=(propensity(x)))

        # ret.append({"X": x, "Y0": mu_0(x), "Y1": mu_1(x)})
        ret["X"].append(x)
        ret["W"].append(w)
        ret["Y0"].append(mu_0(x))
        ret["Y1"].append(mu_1(x))
        if w == 0:
            ret["Y"].append(mu_0(x))
        else:
            ret["Y"].append(mu_1(x))
        # if w == 0:
        #     ret.append({"X": x, "W": w, "Y": mu_0(x)})
        # else:
        #     ret.append({"X": x, "W": w, "Y": mu_1(x)})

    return ret

