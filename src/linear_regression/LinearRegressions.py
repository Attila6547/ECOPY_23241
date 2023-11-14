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

import pandas as pd
import numpy as np
from scipy.stats import t, f

class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.alpha = None
        self.beta = None
        self.p_values = None
        self._model = None

    def fit(self):
        X = np.column_stack([np.ones(len(self.right_hand_side)), self.right_hand_side])
        y = self.left_hand_side
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        alpha = beta[0]
        beta = beta[1:]
        self.alpha = alpha
        self.beta = beta
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self._model = model

    def get_params(self):
        return pd.Series(self._model.params, name='Beta coefficients')

    def get_pvalues(self):
        X = np.column_stack([np.ones(len(self.right_hand_side)), self.right_hand_side])
        y = self.left_hand_side.values
        n, k = X.shape
        df = n - k
        residuals = y - X.dot(np.concatenate(([self.alpha], self.beta)))
        sigma = np.sqrt((residuals.dot(residuals)) / df)
        se = np.sqrt(np.diagonal(sigma ** 2 * np.linalg.inv(X.T.dot(X))))
        t_stats = np.concatenate(([self.alpha / se[0]], self.beta / se[1:]))
        p_values = pd.Series([2 * (1 - stats.t.cdf(np.abs(t), df)) for t in t_stats],name="P-values for the corresponding coefficients")
        self.p_values = p_values
        return p_values

    def get_wald_test_result(self, R):
        wald_value = ((np.array(restriction_matrix).dot(np.concatenate(([self.alpha], self.beta)))) ** 2).sum()
        df_num = np.array(restriction_matrix).shape[0]
        df_denom = len(self.right_hand_side) - len(restriction_matrix)
        p_value = 1 - stats.f.cdf(wald_value, df_num, df_denom)
        result_string = f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        return result_string

    def get_model_goodness_values(self):
        X = np.column_stack([np.ones(len(self.right_hand_side)), self.right_hand_side])
        y = self.left_hand_side
        n, k = X.shape
        df_residuals = n - k
        df_total = n - 1
        y_mean = np.mean(y)
        y_pred = X.dot(np.concatenate(([self.alpha], self.beta)))
        ss_residuals = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - y_mean) ** 2)
        centered_r_squared = 1 - (ss_residuals / ss_total)
        adjusted_r_squared = 1 - (ss_residuals / df_residuals) / (ss_total / df_total)
        return f'Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}'

