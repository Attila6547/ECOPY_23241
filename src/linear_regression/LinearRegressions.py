import pandas as pd
import statsmodels.api as sm
import numpy as np

class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self._model = model

    def get_params(self):
        return pd.Series(self._model.params, name='Beta coefficients')

    def get_pvalues(self):
        return pd.Series(self._model.pvalues, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restriction_matrix):
        restriction_matrix = np.array(restriction_matrix)
        wald_test = self._model.wald_test(restriction_matrix)
        fvalue = round(wald_test.statistic[0, 0], 3)
        pvalue = round(float(wald_test.pvalue), 3)
        return f'F-value: {fvalue}, p-value: {pvalue}'

    def get_model_goodness_values(self):
        ars = round(self._model.rsquared_adj, 3)
        ak = round(self._model.aic, 3)
        by = round(self._model.bic, 3)
        return f'Adjusted R-squared: {ars}, Akaike IC: {ak}, Bayes IC: {by}'






