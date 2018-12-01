# +---------------------------------------------+
# |                                             |
# |  Toolbox para el análisis de las features   |
# |                                             |
# +---------------------------------------------+
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from purgedFolk import PurgedKFold


def featureImportandeMDI (fit, featNames):
    """
    Mean Decrease Impurity. Análisis In Sample (IS) usando árboles de decisión.
    Muy bueno para separar features útiles de las que solo llevan ruido.
    El análisis de las features va uno con otras.
    Se basa en In Sample score reduction.
    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.fom_dict(df0, orinet='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) # because max_features=1

    imp = pd.concat( {'mean': df0.mean(), 'std': df0.std() * df0.shape[0]**-.5}, axis=1)
    imp /= imp['mean'].sum()

    return imp


def featureImportanceMDA (clf, X, y, cv, sample,_weight, t1, pctEmbargo, scoring='neg_log_loss'):
    """
    Mean Decrease Accuracy (MDA) para analizar la importancia de las features.
    Complementa bien al análisis de MDI. Es Out Of Sample (OOS) y vale para 
    cualquier tipo de clasificador, no solo de árbol.
    Ayuda a distinguir las features buenas de las que son ruido.
    Se basa en OOS (out of sample) score reduction.
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('Tipo de scoring no reconocido')
    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbrago=pctEmbargo)
    scr0, scr1 = pd.Series(), pd.DataFrame(columns=X.columns)
