"""Sklearn-style pipeline that threads ``stim_info`` through every step.

Sklearn assumes 2D ``(n_samples, n_features)`` data with samples on axis 0, but here the "sample" axis
(conditions) is axis 1 and the array is 4D with changing dimensionality between steps.
A small custom pipeline is more readable than working around sklearn's validation and
metadata routing.
"""


class IEEGPreprocessingPipeline:
    """Compose preprocessing steps, fitting and transforming them in sequence.

    Parameters
    ----------
    steps : list of (name, step) tuples
        Each ``step`` implements ``fit(X, y, stim_info)`` and
        ``transform(X, stim_info)`` (see ``base.TransformerStep``).
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None, stim_info=None):
        self.fit_transform(X, y, stim_info=stim_info)
        return self

    def transform(self, X, stim_info=None):
        Xt = X
        for _, step in self.steps:
            Xt = step.transform(Xt, stim_info=stim_info)
        return Xt

    def fit_transform(self, X, y=None, stim_info=None):
        Xt = X
        for _, step in self.steps:
            Xt = step.fit_transform(Xt, y, stim_info=stim_info)
        return Xt
