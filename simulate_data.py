import numpy as np


def get_training_set(d, mu_0, mu_1, propensity, count):
    # ret = {"X": [], "W": [], "Y": []}
    # for i in range(count):
    #     x = np.random.normal(loc=0, scale=1, size=d)
    #
    #     w = np.random.binomial(n=1, p=(propensity(x)))
    #
    #     ret["X"].append(x)
    #     ret["W"].append(w)
    #     if w == 0:
    #         ret["Y"].append(mu_0(x))
    #     else:
    #         ret["Y"].append(mu_1(x))

    x = list(map(lambda a: list(a), np.random.normal(loc=0, scale=1, size=(count, d))))

    w = list(map(lambda a: np.random.binomial(n=1, p=(propensity(a))), x))

    y = list(map(lambda a: mu_0(a[0]) if a[1] == 0 else mu_1(a[0]), zip(x, w)))

    ret = {"X": x, "W": w, "Y": y}

    return ret


def get_test_set(d, mu_0, mu_1, propensity, count):
    # ret = {"X": [], "W": [], "Y": [], "Y0": [], "Y1": []}
    # for i in range(count):
    #     x = np.random.normal(loc=0, scale=1, size=d)
    #
    #     w = np.random.binomial(n=1, p=(propensity(x)))
    #
    #     # ret.append({"X": x, "Y0": mu_0(x), "Y1": mu_1(x)})
    #     ret["X"].append(x)
    #     ret["W"].append(w)
    #     ret["Y0"].append(mu_0(x))
    #     ret["Y1"].append(mu_1(x))
    #     if w == 0:
    #         ret["Y"].append(mu_0(x))
    #     else:
    #         ret["Y"].append(mu_1(x))
    #     # if w == 0:
    #     #     ret.append({"X": x, "W": w, "Y": mu_0(x)})
    #     # else:
    #     #     ret.append({"X": x, "W": w, "Y": mu_1(x)})

    x = list(map(lambda a: list(a), np.random.normal(loc=0, scale=1, size=(count, d))))
    w = list(map(lambda a: np.random.binomial(n=1, p=(propensity(a))), x))
    y0 = list(map(lambda a: mu_0(a), x))
    y1 = list(map(lambda a: mu_1(a), x))
    y = list(map(lambda a: a[0] if a[2] == 0 else a[1], zip(y0, y1, w)))

    ret = {"X": x, "W": w, "Y": y, "Y0": y0, "Y1": y1}

    return ret

