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

    def fit(self):
        X = np.column_stack([np.ones(len(self.right_hand_side)), self.right_hand_side])
        y = self.left_hand_side
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def get_params(self):
        return pd.Series({'Beta coefficients': self.beta.tolist()})

    def get_pvalues(self):
        residuals = self.left_hand_side - (self.right_hand_side @ self.beta)
        sigma2 = np.sum(residuals**2) / (len(self.right_hand_side) - 2)
        X = np.column_stack([np.ones(len(self.right_hand_side)), self.right_hand_side])
        XTX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(XTX_inv))
        t_stat = self.beta / se
        p_values = (1 - t.cdf(np.abs(t_stat), df=len(self.right_hand_side) - 2)) * 2
        return pd.Series({'P-values for the corresponding coefficients': p_values})

    def get_wald_test_result(self, R):
        if len(R[0]) != len(self.beta):
            raise ValueError("Matrices are not aligned")
        wald_statistic = ((R @ self.beta) ** 2) / (R @ np.linalg.inv(self.right_hand_side.T @ self.right_hand_side) @ R.T)
        p_value = 1 - t.cdf(wald_statistic, df=len(R), loc=0, scale=1)
        return f'Wald: {wald_statistic:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        y = self.left_hand_side
        y_mean = np.mean(y)
        y_pred = self.right_hand_side @ self.beta
        centered_r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2)
        n = len(self.right_hand_side)
        p = 2
        adjusted_r_squared = 1 - (1 - centered_r_squared) * (n - 1) / (n - p - 1)
        return f'Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}'
















