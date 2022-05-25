import numpy as np


def calc_granger_causality_likelihood_ratio_p(x, maxlag):
    from scipy import stats
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.tsatools import lagmat2ds
    from statsmodels.tools.tools import add_constant

    # Taken from statsmodels.tsa.stattools.grangercausalitytests
    results = np.zeros(maxlag)
    for ind in range(maxlag):
        mlg = ind + 1
        # create lagmat of both time series
        dta = lagmat2ds(x, mlg, trim="both", dropex=1)
        dtaown = add_constant(dta[:, 1 : (mlg + 1)], prepend=False)
        dtajoint = add_constant(dta[:, 1:], prepend=False)

        # Run ols on both models without and with lags of second variable
        res2down = OLS(dta[:, 0], dtaown).fit()
        res2djoint = OLS(dta[:, 0], dtajoint).fit()

        # likelihood ratio test pvalue:
        lr = -2 * (res2down.llf - res2djoint.llf)
        results[ind] = stats.chi2.sf(lr, mlg)

    return results
