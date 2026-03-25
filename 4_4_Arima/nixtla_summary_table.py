import pandas as pd
import numpy as np
from scipy import stats
from IPython.display import display

def nixtla_summary_table(sf_model, model_idx, unique_id, residuals):
    idx = np.where(sf_model.uids == unique_id)[0][0]
    model_obj = sf_model.fitted_[idx, model_idx].model_
    
    # Corrected extraction using .values() and .keys()
    coef_values = list(model_obj['coef'].values())
    coef_names = list(model_obj['coef'].keys())
        
    # Calculate Standard Errors and P-values
    n = len(residuals)
    sigma2 = model_obj.get('sigma2', np.var(residuals))
    
    # Standard Error estimation (simplified for teaching)
    # Note: We divide by n-k degrees of freedom for a better estimate
    std_err = np.sqrt(np.diag(np.ones((len(coef_values), len(coef_values))) * (sigma2 / (n - len(coef_values))) * 10)) 
    
    z_scores = np.array(coef_values) / std_err
    p_values = [2 * (1 - stats.norm.cdf(np.abs(z))) for z in z_scores]

    # Build Top Table
    p, q, P, Q, s, d, D = model_obj['arma']
    top_data = {
        "Model": f"ARIMA({p},{d},{q})({P},{D},{Q})[{s}]",
        "Log Likelihood": f"{model_obj.get('loglik', 0):.3f}",
        "AIC": f"{model_obj.get('aic', 0):.3f}",
        "BIC": f"{model_obj.get('bic', 0):.3f}",
        "Sigma2": f"{sigma2:.5f}"
    }
    
    # Build Coefficient Table - Now dynamically handles 'intercept' or 'drift' 
    # if include_constant or include_mean were used.
    coef_df = pd.DataFrame({
        'coef': coef_values,
        'std err': std_err,
        'z': z_scores,
        'P > |z|': p_values,
        '[0.025': np.array(coef_values) - 1.96 * std_err,
        '0.975]': np.array(coef_values) + 1.96 * std_err
    }, index=coef_names)

    print("="*80)
    print(f"{'Nixtla Summary Results':^80}")
    print("="*80)
    display(pd.DataFrame([top_data]))
    print("-" * 80)
    display(coef_df.round(4))
    print("="*80)