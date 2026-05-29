"""Shared base for pipeline steps.

Every step mimics the sklearn transformer contract but threads a ``stim_info`` dict
alongside the data. Subclasses override ``transform`` (and ``fit`` when they learn
state); the default ``fit`` is a no-op for stateless steps.
"""


class TransformerStep:
    """Mixin providing the fit / fit_transform contract used across the pipeline."""

    def fit(self, X, y=None, stim_info=None):
        return self

    def transform(self, X, stim_info=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None, stim_info=None):
        return self.fit(X, y, stim_info=stim_info).transform(X, stim_info=stim_info)
