import pandas as pd
import pickle
from sklearn.impute import KNNImputer

# SEE BELOW FOR THE USE CASES

def predict_ite_causal_forest(forest_model, data):
    """
    Predicts ITEs using causal forest
    :param forest_model: the causal forest model, created using pickle.load()
    :param data: pandas dataframe with RCT data, created by for example pandas.read_csv("filename", index_col=0)
    :return: Predicted ITEs
    """

    imputed_data = preprocess_data(data)

    return forest_model.predict(X=imputed_data)


def predict_ite_s_learner(s_learner_model, data):
    """
    Predicts ITEs using S-learner
    :param s_learner_model: the S-learner model, created using pickle.load()
    :param data: pandas dataframe with RCT data, created by for example pandas.read_csv("filename", index_col=0)
    :return: Predicted ITEs
    """

    imputed_data = preprocess_data(data)

    return s_learner_model.predict(imputed_data.assign(**{"W": 1})) - s_learner_model.predict(imputed_data.assign(**{"W": 0}))


def predict_ite_t_learner(t_learner_model_0, t_learner_model_1, data):
    """
    Predicts ITEs using S-learner
    :param t_learner_model_0: the T-learner base model for Y_0, created using pickle.load()
    :param t_learner_model_1: the T-learner base model for Y_1, created using pickle.load()
    :param data: pandas dataframe with RCT data, created by for example pandas.read_csv("filename", index_col=0)
    :return: Predicted ITEs
    """

    imputed_data = preprocess_data(data)

    return t_learner_model_1.predict(imputed_data) - t_learner_model_0.predict(imputed_data)


def preprocess_data(data):
    # Drop unnecessary columns and turn everything into numbers
    xs = data.drop(columns=["peep_regime", "mort_28", "id"])
    xs.loc[xs["sex"] == "M", "sex"] = 0
    xs.loc[xs["sex"] == "F", "sex"] = 1
    xs_columns = xs.columns

    # normalise data
    norm_xs = (xs - xs.mean()) / xs.std()

    # Impute missing values
    imputer = KNNImputer()
    imp_xs = imputer.fit_transform(norm_xs)
    imp_xs = pd.DataFrame(data=imp_xs, columns=xs_columns)

    # Select corresponding columns, in the correct order
    selected_columns = ["age", "weight", "pf_ratio", "po2", "pco2", "driving_pressure", "fio2", "minute_volume",
                        "plateau_pressure"]
    imp_xs = imp_xs[selected_columns]

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


# DATA: I'm assuming it is also in a .csv file, with exactly the same columns (also for treatment and outcome)
# If not, then some lines might have to be edited, for example dropping columns on line 49
# All preprocessing steps (normalisation, imputation
# The functions above take a dataframe as input, so we can read the file as
data = pd.read_csv("filename.csv", index_col=0)

# For such loaded models and data we can call:
s_learner_ite = predict_ite_s_learner(s_learner, data)
t_learner_ite = predict_ite_t_learner(t_learner_0, t_learner_1, data)
causal_forest_ite = predict_ite_causal_forest(causal_forest, data)
