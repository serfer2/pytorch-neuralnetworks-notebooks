"""
Clase para trabajar con los sets de datos de las observaciones.
Se trata de purgar los datos quitando los overlapping observations.
Extiende de la clase KFold de scykit
"""
import pandas as pd
from sklearn.model_selection._split import _BaseKFold

class PurgedFolk(_BaseKFold):
    """
    Extends KFold class to work with labels that span intervals.
    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples
    in between.
    """
    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('t1 debe ser instancia de pandas.Series')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, x, y=None, groups=None):
        