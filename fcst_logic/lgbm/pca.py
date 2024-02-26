import logging

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

logger = logging.getLogger("__name__")


def dimensionality_reduction(
    data: tuple[pd.DataFrame, pd.DataFrame],
    columns_to_reduce: list[str],
    threshold_of_variance_explanation: float,
    cv_id: str,
    cluster_id: int,
    name_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_other_columns = [col for col in data[0].columns if col not in columns_to_reduce]
    train_data = data[0][columns_to_reduce]
    test_data = data[1][columns_to_reduce]
    pca = PCA()
    pca.fit(train_data)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    dimensions = np.argmax(cumsum >= threshold_of_variance_explanation) + 1
    logger.info(
        f"""{cv_id}-{cluster_id}: Components that explain at least {threshold_of_variance_explanation*100}% variance of the {len(columns_to_reduce)} columns: {dimensions}"""
    )
    pca = PCA(n_components=dimensions)
    pca.fit(train_data)
    train_pca = pd.DataFrame(pca.transform(train_data), index=train_data.index)
    col_names = [f"FEAT__PCA_{col}_{name_id}" for col in train_pca.columns]
    train_pca.columns = col_names
    test_pca = pd.DataFrame(pca.transform(test_data), index=test_data.index)
    test_pca.columns = col_names
    train_data = pd.concat([data[0][all_other_columns], train_pca], axis=1)
    test_data = pd.concat([data[1][all_other_columns], test_pca], axis=1)
    return train_data, test_data
