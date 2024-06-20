import pandas as pd
import pickle

# SEE BELOW FOR USE EXAMPLES


def predict_ite_causal_forest(forest_model, data, standard_scaler, knn_imputer):
    """
    Predicts ITEs using causal forest
    :param forest_model: the causal forest model, created using pickle.load()
    :param data: pandas dataframe with RCT data, created by for example pandas.read_csv("filename", index_col=0)
    :param standard_scaler: StandardScaler object, to be passed to the preprocessing function
    :param knn_imputer: KNNImputer object, to be passed to the preprocessing function
    :return: Predicted ITEs
    """

    imputed_data = preprocess_data(data, standard_scaler, knn_imputer)

    return forest_model.predict(X=imputed_data)


def predict_ite_s_learner(s_learner_model, data, standard_scaler, knn_imputer):
    """
    Predicts ITEs using S-learner
    :param s_learner_model: the S-learner model, created using pickle.load()
    :param data: pandas dataframe with RCT data, created by for example pandas.read_csv("filename", index_col=0)
    :param standard_scaler: StandardScaler object, to be passed to the preprocessing function
    :param knn_imputer: KNNImputer object, to be passed to the preprocessing function
    :return: Predicted ITEs
    """

    imputed_data = preprocess_data(data, standard_scaler, knn_imputer)

    return s_learner_model.predict(imputed_data.assign(**{"W": 1})) - s_learner_model.predict(imputed_data.assign(**{"W": 0}))


def predict_ite_t_learner(t_learner_model_0, t_learner_model_1, data, standard_scaler, knn_imputer):
    """
    Predicts ITEs using S-learner
    :param t_learner_model_0: the T-learner base model for Y_0, created using pickle.load()
    :param t_learner_model_1: the T-learner base model for Y_1, created using pickle.load()
    :param data: pandas dataframe with RCT data, created by for example pandas.read_csv("filename", index_col=0)
    :param standard_scaler: StandardScaler object, to be passed to the preprocessing function
    :param knn_imputer: KNNImputer object, to be passed to the preprocessing function
    :return: Predicted ITEs
    """

    imputed_data = preprocess_data(data, standard_scaler, knn_imputer)

    return t_learner_model_1.predict(imputed_data) - t_learner_model_0.predict(imputed_data)


def preprocess_data(data, standard_scaler, knn_imputer):
    # Select corresponding columns, in the correct order
    selected_columns = ["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "fio2", "minute_volume",
                        "plateau_pressure"]
    xs = data[selected_columns]
    xs_columns = xs.columns

    # normalise data
    norm_xs = standard_scaler.transform(X=xs)

    # Impute missing values
    imp_xs = knn_imputer.transform(X=norm_xs)
    imp_xs = pd.DataFrame(data=imp_xs, columns=xs_columns)

    return imp_xs



# MODELS: all models can be read using pickle with the following code

with open("causal_forest_model.pkl", "rb") as f:
    causal_forest = pickle.load(f)

with open("s_learner_model.pkl", "rb") as f:
    s_learner = pickle.load(f)

with open("t_learner_model_0.pkl", "rb") as f:
    t_learner_0 = pickle.load(f)

with open("t_learner_model_1.pkl", "rb") as f:
    t_learner_1 = pickle.load(f)

with open("knn_imputer.pkl", "rb") as f:
    knn_imputer = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    standard_scaler = pickle.load(f)


# DATA: I'm assuming it is also in a .csv file, with exactly the same columns
# The functions above take a dataframe as input, so we can read the file as
data = pd.read_csv("filename.csv", index_col=0)

# For such loaded models and data we can call:
s_learner_ite = predict_ite_s_learner(s_learner, data, standard_scaler, knn_imputer)
t_learner_ite = predict_ite_t_learner(t_learner_0, t_learner_1, data, standard_scaler, knn_imputer)
causal_forest_ite = predict_ite_causal_forest(causal_forest, data, standard_scaler, knn_imputer)
