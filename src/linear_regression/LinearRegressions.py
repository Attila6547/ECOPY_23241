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


import pandas as pd
import statsmodels.api as sm
import numpy as np
import scipy

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = sm.add_constant(self.right_hand_side)
        self._model = None

    def fit(self):
        _model = sm.GLS(self.left_hand_side, self.right_hand_side).fit()

    def get_params(self):
        self.a = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.b = np.dot(np.transpose(self.right_hand_side), self.left_hand_side)
        self.params = np.dot(np.linalg.inv(self.a), self.b)
        self.resid = self.left_hand_side - np.dot(self.right_hand_side, self.params)

        self.resid_sq = self.resid ** 2
        self.log_resid_sq = np.log(self.resid_sq)
        self.a_residsq_regr = np.dot(np.transpose(self.right_hand_side), self.right_hand_side)
        self.b_residsq_regr = np.dot(np.transpose(self.right_hand_side),
                                     self.log_resid_sq)
        self.params_residsq_regr = np.dot(np.linalg.inv(self.a_residsq_regr), self.b_residsq_regr)

        self.resid_3rd = np.dot(self.right_hand_side, self.params_residsq_regr)
        self.resid_3rd_unlogged = np.sqrt(np.exp(self.resid_3rd))
        self.resid_3rd_unlogged_inv = 1 / self.resid_3rd_unlogged
        self.Vinv = (np.diag(self.resid_3rd_unlogged_inv))
        self.FGLS_a = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.right_hand_side)
        self.FGLS_b = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.left_hand_side)

        self.FGLS_params = np.dot(np.linalg.inv(self.FGLS_a), self.FGLS_b)

        return pd.Series(self.FGLS_params, name='Beta coefficients')

    def get_pvalues(self):
        self.get_params()
        self.a_inv = np.linalg.inv(self.FGLS_a)
        self.errors = self.left_hand_side - np.dot(self.right_hand_side, self.FGLS_params)
        self.n = len(self.left_hand_side)
        self.p = len(self.right_hand_side.columns)
        self.var = np.dot(np.transpose(self.errors), self.errors) / (self.n - self.p)
        self.se_sq = self.var * np.diag(self.a_inv)
        self.se = np.sqrt(self.se_sq)
        self.t_stats = np.divide(self.FGLS_params, self.se)
        term = np.minimum(scipy.stats.t.cdf(self.t_stats, self.n - self.p),
                          1 - scipy.stats.t.cdf(self.t_stats, self.n - self.p))
        p_values = (term) * 2
        return pd.Series(p_values, name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, restr_matrix):
        self.get_params()
        self.get_pvalues()
        r = np.transpose(np.zeros((len(restr_matrix))))
        term_1 = np.dot(restr_matrix, self.FGLS_params) - r
        term_2 = np.dot(np.dot(restr_matrix, self.a_inv), np.transpose(restr_matrix))
        f_stat = (np.dot(np.transpose(term_1), np.dot(np.linalg.inv(term_2), term_1)) / len(restr_matrix)) / self.var
        p_value = (1 - scipy.stats.f.cdf(f_stat, len(restr_matrix), self.n - self.p))
        f_stat.astype(float)
        p_value.astype(float)
        return f'Wald: {f_stat:.3f}, p-value: {p_value:.3f}'

    def get_model_goodness_values(self):
        self.get_params()
        self.get_pvalues()
        self.errors = self.left_hand_side - np.dot(self.right_hand_side, self.FGLS_params)

        y_demean = self.left_hand_side
        w = np.dot(np.dot(np.transpose(y_demean), self.Vinv), y_demean)
        SSE_1 = np.dot(np.dot(np.transpose(self.left_hand_side), self.Vinv), self.right_hand_side)
        SSE_2 = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.right_hand_side)
        SSE_3 = np.dot(np.dot(np.transpose(self.right_hand_side), self.Vinv), self.left_hand_side)
        SSE = np.dot(np.dot(SSE_1, np.linalg.inv(SSE_2)), np.transpose(SSE_1))
        SST = w
        r2 = 1 - SSE / SST
        adj_r2 = 1 - (self.n - 1) / (self.n - self.p) * (1 - r2)
        return f'Centered R-squared: {r2:.3f}, Adjusted R-squared: {adj_r2:.3f}'


import pandas as pd
from scipy.optimize import minimize
import numpy as np
from statsmodels.tools.tools import add_constant

class LinearRegressionML:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = add_constant(self.right_hand_side)

        self._params = None

    def fit(self):
        def neg_log_likelihood(params, X, y):
            predicted = np.dot(X, params)
            log_likelihood = -0.5 * (np.log(2 * np.pi * np.var(y)) + np.sum((y - predicted) ** 2) / np.var(y))
            return -1 * log_likelihood

        initial_params = np.zeros(self.right_hand_side.shape[1]) + 0.1

        result = minimize(neg_log_likelihood, initial_params, args=(self.right_hand_side, self.left_hand_side), method='L-BFGS-B')

        if result.success:
            self._params = result.x
        else:
            raise ValueError("MLE fit failed!")

    def get_params(self):
        if self._params is not None:
            return pd.Series(self._params, index=self.right_hand_side.columns, name='Beta coefficients')
        else:
            raise ValueError("Fit the model first!")

    def get_pvalues(self):
        if self._params is not None:
            X = add_constant(self.right_hand_side.iloc[:, 1:]) if 'const' not in self.right_hand_side.columns else self.right_hand_side

            residuals = self.left_hand_side - np.dot(X, self._params)
            n = len(self.left_hand_side)
            p = X.shape[1]
            var = np.sum(residuals ** 2) / (n - p)
            cov_matrix = var * np.linalg.inv(np.dot(X.T, X))

            se = np.sqrt(np.diag(cov_matrix))
            t_stats = self._params / se

            p_values = [min(t.cdf(-np.abs(t_stat), df=n-p)*2, t.cdf(np.abs(t_stat), df=n-p)*2) for t_stat in t_stats]

            return pd.Series(p_values, index=X.columns, name='P-values for the corresponding coefficients')
        else:
            raise ValueError("Fit the model first!")

    def get_model_goodness_values(self):
        if self._params is not None:
            residuals = self.left_hand_side - np.dot(self.right_hand_side, self._params)
            SSE = np.sum(residuals ** 2)
            SST = np.sum((self.left_hand_side - np.mean(self.left_hand_side)) ** 2)
            r_squared = 1 - (SSE / SST)
            adj_r_squared = 1 - (1 - r_squared) * (len(self.left_hand_side) - 1) / (len(self.left_hand_side) - len(self.right_hand_side.columns))
            return f'Centered R-squared: {r_squared:.3f}, Adjusted R-squared: {adj_r_squared:.3f}'
        else:
            raise ValueError("Fit the model first!")



