import numpy as np
import pandas as pd
import random
import simulate_data
from matplotlib import pyplot as plt
from scipy.stats import beta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from econml.grf import CausalForest


def cumulative_gain(dataset, prediction, y, t, min_periods=30, steps=100):
    size = dataset.shape[0]
    ordered_df = dataset.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_periods, size, size // steps)) + [size]
    return np.array([elast(ordered_df.head(rows), y, t) * (rows/size) for rows in n_rows])


def elast(data, y, t):
    return (np.sum((data[t] - data[t].mean())*(data[y] - data[y].mean())) /
            np.sum((data[t] - data[t].mean())**2))


def train_and_evaluate(train, test, train_x_t_0_head, train_x_t_1_head):

    train_x = pd.DataFrame(train["X"].tolist(), index=train.index)
    train_x.columns = train_x.columns.astype(str)
    train_y = train["Y"]
    train_w = train["W"]

    train_x_s = train_x.assign(W=pd.Series(train_w).values)
    train_x_t_0 = pd.DataFrame(train.query("W==0")["X"].tolist())
    train_x_t_0.columns = train_x_t_0.columns.astype(str)
    train_y_t_0 = train.query("W==0")["Y"]
    if train_x_t_0.shape[0] == 0:
        row = train_x_t_0_head
        train_x_t_0 = pd.DataFrame(row["X"].tolist())
        train_x_t_0.columns = train_x_t_0.columns.astype(str)
        train_y_t_0 = row["Y"]

    train_x_t_1 = pd.DataFrame(train.query("W==1")["X"].tolist())
    train_x_t_1.columns = train_x_t_1.columns.astype(str)
    train_y_t_1 = train.query("W==1")["Y"]
    if train_x_t_1.shape[0] == 0:
        row = train_x_t_1_head
        train_x_t_1 = pd.DataFrame(row["X"].tolist())
        train_x_t_1.columns = train_x_t_1.columns.astype(str)
        train_y_t_1 = row["Y"]

    test_x = pd.DataFrame(test["X"].tolist(), index=test.index)
    test_x.columns = test_x.columns.astype(str)

    true_test_cate = test.Y1 - test.Y0
    # print("COPIED TABLES")

    # S-learner
    regr = RandomForestRegressor()
    regr.fit(train_x_s, train_y)

    s_learner_cate = regr.predict(test_x.assign(**{"W": 1})) - regr.predict(test_x.assign(**{"W": 0}))

    mse_s = mean_squared_error(s_learner_cate, true_test_cate)

    # print("TRAINED S-LEARNER")

    # T-learner
    regr0 = RandomForestRegressor()
    regr1 = RandomForestRegressor()
    regr0.fit(train_x_t_0, train_y_t_0)
    regr1.fit(train_x_t_1, train_y_t_1)

    t_learner_cate = regr1.predict(test_x) - regr0.predict(test_x)
    mse_t = mean_squared_error(t_learner_cate, true_test_cate)
    # print("TRAINED T-LEARNER")

    # Causal forest
    cau_forest = CausalForest()
    cau_forest.fit(train_x, train_w, train_y)

    causal_forest_cate = cau_forest.predict(test_x)
    mse_forest = mean_squared_error(causal_forest_cate, true_test_cate)

    # K Nearest Neighbours
    knn_0 = KNeighborsRegressor(n_neighbors=min(10, train_x_t_0.shape[0]))
    knn_1 = KNeighborsRegressor(n_neighbors=min(10, train_x_t_1.shape[0]))
    knn_0.fit(train_x_t_0, train_y_t_0)
    knn_1.fit(train_x_t_1, train_y_t_1)

    knn_cate = knn_1.predict(test_x) - knn_0.predict(test_x)
    mse_knn = mean_squared_error(knn_cate, true_test_cate)

    return mse_s, mse_t, mse_forest, mse_knn


def run_test(d, mu_0, mu_1, e, n_train, n_test, reps):
    t_s = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    training_sizes = [int(n_train * i) for i in t_s]
    mse_s_list = [[] for _ in training_sizes]
    mse_t_list = [[] for _ in training_sizes]
    mse_forest_list = [[] for _ in training_sizes]
    mse_knn_list = [[] for _ in training_sizes]

    for r in range(reps):

        train_full = pd.DataFrame(simulate_data.get_training_set(d, mu_0, mu_1, e, n_train))
        test_full = pd.DataFrame(simulate_data.get_test_set(d, mu_0, mu_1, e, n_test))

        for i, size in enumerate(training_sizes):

            train = train_full.head(size)
            test = test_full

            mse_s, mse_t, mse_forest, mse_knn = train_and_evaluate(train, test, train_full.query("W==0").head(1), train_full.query("W==1").head(1))

            mse_s_list[i].append(mse_s)

            mse_t_list[i].append(mse_t)

            mse_forest_list[i].append(mse_forest)

            mse_knn_list[i].append(mse_knn)

    return training_sizes, mse_s_list, mse_t_list, mse_forest_list, mse_knn_list


def plot_mse(training_sizes, mse_s_list, mse_t_list, mse_forest_list, mse_knn_list):
    mse_s_final = list(map(lambda l: np.mean(l), mse_s_list))
    mse_t_final = list(map(lambda l: np.mean(l), mse_t_list))
    mse_forest_final = list(map(lambda l: np.mean(l), mse_forest_list))
    mse_knn_final = list(map(lambda l: np.mean(l), mse_knn_list))

    plt.plot(training_sizes, mse_s_final, label="S-learner")
    plt.plot(training_sizes, mse_t_final, label="T-learner")
    plt.plot(training_sizes, mse_forest_final, label="Causal forest")
    plt.plot(training_sizes, mse_knn_final, label="KNN")
    plt.ylabel("MSE")
    plt.xlabel("Training size")
    plt.legend()
    plt.show()


def get_conf_intervals(mse_s_list, mse_t_list, mse_forest_list, mse_knn_list):
    mse_s_all = mse_s_list[-1]
    mse_t_all = mse_t_list[-1]
    mse_forest_all = mse_forest_list[-1]
    mse_knn_all = mse_knn_list[-1]

    data = {"S-Learner": [np.mean(mse_s_all), np.std(mse_s_all, ddof=1)], "T-Learner": [np.mean(mse_t_all), np.std(mse_t_all, ddof=1)], "Causal Forest": [np.mean(mse_forest_all), np.std(mse_forest_all, ddof=1)], "KNN": [np.mean(mse_knn_all), np.std(mse_knn_all, ddof=1)] }

    return pd.DataFrame.from_dict(data, orient='index', columns=["Mean", "Std dev"])


def plot_gain_curves(d, mu_0, mu_1, e, n_train, n_test):
    train = pd.DataFrame(simulate_data.get_training_set(d, mu_0, mu_1, e, n_train))
    test = pd.DataFrame(simulate_data.get_test_set(d, mu_0, mu_1, e, n_test))

    train_x = pd.DataFrame(train["X"].tolist(), index = train.index)
    train_x.columns = train_x.columns.astype(str)
    train_y = train["Y"]
    train_w = train["W"]

    train_x_s = train_x.assign(W=pd.Series(train_w).values)

    train_x_t_0 = pd.DataFrame(train.query("W==0")["X"].tolist())
    train_x_t_0.columns = train_x_t_0.columns.astype(str)
    train_y_t_0 = train.query("W==0")["Y"]

    train_x_t_1 = pd.DataFrame(train.query("W==1")["X"].tolist())
    train_x_t_1.columns = train_x_t_1.columns.astype(str)
    train_y_t_1 = train.query("W==1")["Y"]

    test_x = pd.DataFrame(test["X"].tolist(), index=test.index)
    test_x.columns = test_x.columns.astype(str)

    # S-learner
    regr_s = RandomForestRegressor()
    regr_s.fit(train_x_s, train_y)

    # T-learner
    regr0_t = RandomForestRegressor()
    regr1_t = RandomForestRegressor()
    regr0_t.fit(train_x_t_0, train_y_t_0)
    regr1_t.fit(train_x_t_1, train_y_t_1)

    # Causal forest
    cau_forest = CausalForest()
    cau_forest.fit(train_x, train["W"], train["Y"])

    # KNN
    knn_0 = KNeighborsRegressor(n_neighbors=10)
    knn_1 = KNeighborsRegressor(n_neighbors=10)
    knn_0.fit(train_x_t_0, train_y_t_0)
    knn_1.fit(train_x_t_1, train_y_t_1)

    # GET CATE

    s_learner_cate_train = train_x.assign(cate=regr_s.predict(train_x.assign(**{"W": 1})) - regr_s.predict(train_x.assign(**{"W": 0})), W=train_w, Y=train_y)
    s_learner_cate_test = test_x.assign(cate=regr_s.predict(test_x.assign(**{"W": 1})) - regr_s.predict(test_x.assign(**{"W": 0})), W=test["W"], Y=test["Y"])

    t_learner_cate_train = train_x.assign(cate=regr1_t.predict(train_x) - regr0_t.predict(train_x), W=train_w, Y=train_y)
    t_learner_cate_test = test_x.assign(cate=regr1_t.predict(test_x) - regr0_t.predict(test_x), W=test["W"], Y=test["Y"])

    causal_forest_cate_train = train_x.assign(cate=cau_forest.predict(train_x), W=train_w, Y=train_y)
    causal_forest_cate_test = test_x.assign(cate=cau_forest.predict(test_x), W=test["W"], Y=test["Y"])

    knn_cate_train = train_x.assign(cate=knn_1.predict(train_x) - knn_0.predict(train_x), W=train_w, Y=train_y)
    knn_cate_test = test_x.assign(cate=knn_1.predict(test_x) - knn_0.predict(test_x), W=test["W"], Y=test["Y"])

    # GET GAIN CURVES

    gain_curve_train_s = cumulative_gain(s_learner_cate_train, "cate", y="Y", t="W")
    gain_curve_test_s = cumulative_gain(s_learner_cate_test, "cate", y="Y", t="W")

    gain_curve_train_t = cumulative_gain(t_learner_cate_train, "cate", y="Y", t="W")
    gain_curve_test_t = cumulative_gain(t_learner_cate_test, "cate", y="Y", t="W")

    gain_curve_train_c_f = cumulative_gain(causal_forest_cate_train, "cate", y="Y", t="W")
    gain_curve_test_c_f = cumulative_gain(causal_forest_cate_test, "cate", y="Y", t="W")

    gain_curve_train_knn = cumulative_gain(knn_cate_train, "cate", y="Y", t="W")
    gain_curve_test_knn = cumulative_gain(knn_cate_test, "cate", y="Y", t="W")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))

    ax1.plot(gain_curve_train_s, color="C1", label="Train")
    ax1.plot(gain_curve_test_s, color="C0", label="Test")
    ax1.plot([0, 100], [0, elast(test, "Y", "W")], linestyle="--", color="black", label="Baseline")
    ax1.legend()
    ax1.set_title("Cumulative gain curve - S-learner")

    ax2.plot(gain_curve_train_t, color="C1", label="Train")
    ax2.plot(gain_curve_test_t, color="C0", label="Test")
    ax2.plot([0, 100], [0, elast(test, "Y", "W")], linestyle="--", color="black", label="Baseline")
    ax2.legend()
    ax2.set_title("Cumulative gain curve - T-learner")

    ax3.plot(gain_curve_train_c_f, color="C1", label="Train")
    ax3.plot(gain_curve_test_c_f, color="C0", label="Test")
    ax3.plot([0, 100], [0, elast(test, "Y", "W")], linestyle="--", color="black", label="Baseline")
    ax3.legend()
    ax3.set_title("Cumulative gain curve - Causal forest")

    ax4.plot(gain_curve_train_knn, color="C1", label="Train")
    ax4.plot(gain_curve_test_knn, color="C0", label="Test")
    ax4.plot([0, 100], [0, elast(test, "Y", "W")], linestyle="--", color="black", label="Baseline")
    ax4.legend()
    ax4.set_title("Cumulative gain curve - KNN")
