"""
Clase para trabajar con los sets de datos de las observaciones.
Se trata de purgar los datos quitando los overlapping observations.
Extiende de la clase KFold de scykit
"""
import pandas as pd
import numpy as np
from sklearn.model_selection._split import _BaseKFold

class PurgedKFold(_BaseKFold):
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


    def split(self, X, y=None, groups=None):
        
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')
        
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [ (i[0], i[-1]+1) for i in np.array_split( np.arange(X.shape[0]), self.n_splits ) ]

        for i, j in test_starts:

            t0 = self.t1.index[i] # start of test set
            test_indices = indices[i:j]
            maxT1Idx = self.t1.index.searchsorted( self.t1[test_indices].max() )
            train_indices = self.t1.index.searchsorted( self.t1[self.t1 <= t0].index )

            if maxT1Idx < X.shape[0]:
                train_indices = np.concatenate( (train_indices, indices[maxT1Idx + mbrg]) )

            yield train_indices, test_indices