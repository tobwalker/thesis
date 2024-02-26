import logging

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger("__name__")


def tuner(
    data: tuple[pd.DataFrame, pd.DataFrame],
    target_column: str,
    n_estimators: list[int],
    learning_rate: list[float],
    max_depth: list[int],
    num_leaves: list[int],
    n_jobs: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[0]
    test = data[1]
    test_pass_on = data[1][["FEAT__HORIZON", f"TARGET__{target_column}"]].copy()
    train_x = train[[col for col in train.columns if col.startswith("FEAT__")]]
    train_y = train[[col for col in train.columns if col.startswith("TARGET__")]]
    test_x = test[[col for col in test.columns if col.startswith("FEAT__")]]
    test_y = test[[col for col in test.columns if col.startswith("TARGET__")]]

    if (
        (np.isinf(train_x.values).any())
        | (np.isinf(train_y.values).any())
        | (np.isinf(test_x.values).any())
        | (np.isinf(test_y.values).any())
    ):
        raise ValueError("Data that is pushed in LGBM model for training/testing contains infinite values")

    for ne in n_estimators:
        for lr in learning_rate:
            for md in max_depth:
                for nl in num_leaves:
                    lgbm = lgb.LGBMRegressor(
                        n_estimators=ne,
                        learning_rate=lr,
                        max_depth=md,
                        num_leaves=nl,
                        n_jobs=n_jobs,
                    )
                    lgbm.fit(train_x, train_y)
                    test_pred_y = lgbm.predict(test_x)
                    test_pass_on = pd.concat(
                        [
                            test_pass_on,
                            pd.DataFrame(
                                {f"PRED__NE_{ne}_LR_{lr}_MD_{md}_NL_{nl}": test_pred_y}, index=test_pass_on.index
                            ),
                        ],
                        axis=1,
                    )
    return test_pass_on
