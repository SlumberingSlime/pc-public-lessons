# BLOCK 1
import ssl
ssl._create_default_https_context=ssl._create_unverified_context

# BLOCK 2
import numpy as np
import pandas as pd

from functools import partial
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)


def load_mtpl2(n_samples=None):
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]

def plot_obs_pred(
    df,
    feature,
    weight,
    observed,
    predicted,
    y_label=None,
    title=None,
    ax=None,
    fill_legend=False,
):
    df_ = df.loc[:, [feature, weight]].copy()
    df_["observed"] = df[observed] * df[weight]
    df_["predicted"] = predicted * df[weight]
    df_ = (
        df_.groupby([feature])[[weight, "observed", "predicted"]]
        .sum()
        .assign(observed=lambda x: x["observed"] / x[weight])
        .assign(predicted=lambda x: x["predicted"] / x[weight])
    )

    ax = df_.loc[:, ["observed", "predicted"]].plot(style=".", ax=ax)
    y_max = df_.loc[:, ["observed", "predicted"]].values.max() * 0.8
    p2 = ax.fill_between(
        df_.index,
        0,
        y_max * df_[weight] / df_[weight].values.max(),
        color="g",
        alpha=0.1,
    )
    if fill_legend:
        ax.legend([p2], ["{} distribution".format(feature)])
    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: Observed vs Predicted",
    )


def score_estimator(
    estimator,
    X_train,
    X_test,
    df_train,
    df_test,
    target,
    weights,
    tweedie_powers=None,
):
    """Evaluate an estimator on train and test sets with different metrics"""

    metrics = [
        ("D² explained", None),  # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
    ]
    if tweedie_powers:
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )
            for power in tweedie_powers
        ]

    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y, _weights = df[target], df[weights]
        for score_label, metric in metrics:
            if isinstance(estimator, tuple) and len(estimator) == 2:
                # Score the model consisting of the product of frequency and
                # severity models.
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:
                y_pred = estimator.predict(X)

            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res.append({"subset": subset_label, "metric": score_label, "score": score})

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ["train", "test"]]
    )
    return res

# BLOCK 3
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

df = load_mtpl2()

df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
df["Exposure"] = df["Exposure"].clip(upper=1)
df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)

log_scale_transformer = make_pipeline(
    FunctionTransformer(func=np.log), StandardScaler()
)

column_trans = ColumnTransformer(
    [
        (
            "binned_numeric",
            KBinsDiscretizer(n_bins=10, subsample=int(2e5), random_state=0),
            ["VehAge", "DrivAge"],
        ),
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
        ),
        ("passthrough_numeric", "passthrough", ["BonusMalus"]),
        ("log_scaled_numeric", log_scale_transformer, ["Density"]),
    ],
    remainder="drop",
)
X = column_trans.fit_transform(df)

df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

df["Frequency"] = df["ClaimNb"] / df["Exposure"]
df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)

with pd.option_context("display.max_columns", 15):
    print(df[df.ClaimAmount > 0].head())

# BLOCK 4
from sklearn.model_selection import train_test_split

df_train, df_test, X_train, X_test = train_test_split(df, X, random_state=0)

# BLOCK 5
from sklearn.linear_model import TweedieRegressor

glm_pure_premium = TweedieRegressor(power=1.9, alpha=0.1, solver='newton-cholesky')
glm_pure_premium.fit(
    X_train, df_train["PurePremium"], sample_weight=df_train["Exposure"]
)

tweedie_powers = [1.5, 1.7, 1.8, 1.9, 1.99, 1.999, 1.9999]

scores_glm_pure_premium = score_estimator(
    glm_pure_premium,
    X_train,
    X_test,
    df_train,
    df_test,
    target="PurePremium",
    weights="Exposure",
    tweedie_powers=tweedie_powers,
)

scores = pd.concat(
    [scores_glm_pure_premium],
    axis=1,
    sort=True,
    keys=("TweedieRegressor"),
)
print("Evaluation of the Product Model and the Tweedie Regressor on target PurePremium")
with pd.option_context("display.expand_frame_repr", False):
    print(scores)

# BLOCK 6
n_iter = glm_pure_premium.n_iter_

# BLOCK 7
import secretflow as sf

print('The version of SecretFlow: {}'.format(sf.__version__))

sf.shutdown()

sf.init(['alice', 'bob'], address='local')

alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(
    sf.utils.testing.cluster_def(
        ['alice', 'bob'],
        {"protocol": "REF2K", "field": "FM128", "fxp_fraction_bits": 40},
    ),
)

# BLOCK 8
from secretflow.data import FedNdarray, PartitionWay

x, y = X_train, df_train["PurePremium"]
w = df_train["Exposure"]


def x_to_vdata(x):
    x = x.todense()
    v_data = FedNdarray(
        partitions={
            alice: alice(lambda: x[:, :15])(),
            bob: bob(lambda: x[:, 15:])(),
        },
        partition_way=PartitionWay.VERTICAL,
    )
    return v_data


v_data = x_to_vdata(x)

label_data = FedNdarray(
    partitions={alice: alice(lambda: y.values)()},
    partition_way=PartitionWay.VERTICAL,
)

sample_weight = FedNdarray(
    partitions={alice: alice(lambda: w.values)()},
    partition_way=PartitionWay.VERTICAL,
)

# BLOCK 9
from secretflow.device.driver import reveal
from secretflow.ml.linear.ss_glm.core import get_dist

dist = 'Tweedie'
ss_glm_power = 1.9


class DirectRevealModel:
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, X):
        vdata = x_to_vdata(X)
        y = self.model.predict(vdata)
        return reveal(y).reshape((-1,))

    def score(self, X, y, sample_weight=None):
        y = y.values
        y_pred = self.predict(X)

        constant = np.mean(y)
        if sample_weight is not None:
            constant *= sample_weight.shape[0] / np.sum(sample_weight)

        # Missing factor of 2 in deviance cancels out.
        deviance = get_dist(dist, 1, ss_glm_power).deviance(y_pred, y, None)
        deviance_null = get_dist(dist, 1, ss_glm_power).deviance(
            np.average(y, weights=sample_weight) + np.zeros(y.shape), y, None
        )
        return 1 - (deviance + constant) / (deviance_null + constant)

# BLOCK 10
import time
from secretflow.ml.linear.ss_glm import SSGLM

model = SSGLM(spu)

ss_glm_power = 1.9
start = time.time()
model.fit_irls(
    v_data,
    label_data,
    None,
    sample_weight,
    2,
    'Log',
    'Tweedie',
    ss_glm_power,
    l2_lambda=0.1,
    infeed_batch_size_limit=10000000,
    fraction_of_validation_set=0.2,
    stopping_rounds=2,
    stopping_metric='deviance',
    stopping_tolerance=0.001,
)

wrapped_model = DirectRevealModel(model)

# BLOCK 11
reveal(model.spu_w)


# BLOCK 12
tweedie_powers = [1.5, 1.7, 1.8, 1.9, 1.99, 1.999, 1.9999]

scores_ss_glm_pure_premium = score_estimator(
    wrapped_model,
    X_train,
    X_test,
    df_train,
    df_test,
    target="PurePremium",
    weights="Exposure",
    tweedie_powers=tweedie_powers,
)

# BLOCK 13
scores = pd.concat(
    [scores_glm_pure_premium, scores_ss_glm_pure_premium],
    axis=1,
    sort=True,
    keys=("TweedieRegressor", "SSGLMRegressor"),
)
print("Evaluation of the Tweedie Regressor and SS GLM on target PurePremium")
with pd.option_context("display.expand_frame_repr", False):
    print(scores)
    # print(scores.shape) # (10, 4)
    print(scores.columns)
    # print(scores['TweedieRegressor', 'train'])
    train_plain = scores['TweedieRegressor', 'train']
    train_cipher = scores['SSGLMRegressor', 'train']
    train_cmp = (train_cipher - train_plain) / train_plain
    print("明密文训练数据比较\n", train_cmp)
    test_plain = scores['TweedieRegressor', 'test']
    test_cipher = scores['SSGLMRegressor', 'test']
    train_cmp = (test_cipher - test_plain) / test_plain
    print("明密文测试数据比较\n", train_cmp)

# BLOCK 14
res = []
for subset_label, x, df in [
    ("train", X_train, df_train),
    ("test", X_test, df_test),
]:
    exposure = df["Exposure"].values
    res.append(
        {
            "subset": subset_label,
            "observed": df["ClaimAmount"].values.sum(),
            "predicted, tweedie, power=%.2f"
            % glm_pure_premium.power: np.sum(exposure * glm_pure_premium.predict(x)),
            "predicted, ss glm, power=%.2f"
            % ss_glm_power: np.sum(exposure * wrapped_model.predict(x)),
        }
    )

print(pd.DataFrame(res).set_index("subset").T)

# BLOCKS END