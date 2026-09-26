"""
ml package initialization.
Provides compatibility patch for scikit-learn unpickling of SimpleImputer.
"""
try:
    from sklearn.impute import SimpleImputer
    if not hasattr(SimpleImputer, '_fill_dtype'):
        def _get_fill_dtype(self):
            return getattr(self, '_fit_dtype', None)
        def _set_fill_dtype(self, val):
            self._fit_dtype = val
        SimpleImputer._fill_dtype = property(_get_fill_dtype, _set_fill_dtype)
except Exception:
    pass
