# import itertools
# import warnings
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima_process import ArmaProcess
# 
# from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import pmdarima as pmd
# 
# import rdatasets

from tqdm import tqdm


plt.rc('axes', titlesize='medium')
plt.rc('axes', titlelocation='left')
plt.rc('axes.spines', right=False)
plt.rc('axes.spines', top=False)
sizets = (8,4.5)
plt.rc('figure', figsize=sizets)


def summarize(gb, f):
    """Summarize grouped things."""
    return gb.apply(lambda x: pd.Series(f(x)))


def compute(df, f):
    """Compute new (or replacement) columns."""
    newdf = pd.DataFrame(f(df), index=df.index)
    dropcols = [col for col in newdf.columns if col in df.columns]
    if dropcols:
        df = df.drop(columns=dropcols)
    return df.join(newdf)


def set_freq(df, freq=None):
    """Set frequency of DateTimeIndex."""
    if freq is None:
        freq = pd.infer_freq(df.index)
    return df.asfreq(freq)


def extend_timeseries(df, tmax=None, tmin=None, dt=None):
    """Extend timeseries data to later or earlier times."""
    freq = df.index.freq
    if tmax is tmin is dt is None:
        dt = 1
    if tmin is None:
        tmin = df.index.min()
    if tmax is None:
        tmax = df.index.max()
    if dt is not None:
        if isinstance(dt, int):
            if dt > 0:
                tmax += dt * freq
            elif dt < 0:
                tmin += dt * freq
        else:
            dt = pd.to_timedelta(dt)
            if dt > pd.to_timedelta('0d'):
                tmax += dt
            else:
                tmin -= dt
    index = pd.date_range(tmin, tmax, freq=freq)
    return df.reindex(index)


def suptitle(fig, text=None, **kw):
    """Add a nice left-aligned suptitle."""
    if text is None:
        fig, text = plt.gcf(), fig
    fig = fig.figure or fig
    fig.text(fig.subplotpars.left, .99, text, ha='left', va='top', size='large', **kw)
    
    
def rlabel(ax, label=None, **kw):
    """Add a right-side axis title."""
    if label is None:
        ax, label = plt.gca(), ax
    bbox = kw.pop('bbox', dict(facecolor='.9', alpha=0.2))
    ax.text(1, .5, label,
            rotation=-90, ha='left', va='center', transform=ax.transAxes,
            bbox=bbox, **kw)
    

def xdate(ax, fmt=None, freq=None):
    """Tweak x-axis date formatting."""
    dates = plt.matplotlib.dates
    if fmt is None:
        ax, fmt = plt.gca(), ax
    if freq:
        t1, t2 = dates.num2date(ax.get_xticks()[[0,-1]])
        ticks = pd.date_range(t1, t2, freq=freq)
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(dates.DateFormatter(fmt))
    
    
def plot_tsresiduals(Y, y, acf_lags=np.r_[1:26]):
    """Plot timeseries residuals for ground truth Y and estimate y."""
    fig = plt.figure()
    gs = plt.GridSpec(3, 2, figure=fig)
    ts_ax = fig.add_subplot(gs[0,:])
    axs = np.array([ts_ax] + [fig.add_subplot(gs[i,j]) for j in (0,1) for i in (1,2)])
    ax, rax, hax, acfax, pacfax = axs
    #((ax, hax), (rax, acfax)) = axs
    mask = ~(np.isnan(Y) | np.isnan(y))
    Y, y = Y[mask], y[mask]
    #dy = y - Y
    # I was surprised by this convention but ok
    dy = Y - y
    ax.plot(Y, color='k')
    ax.plot(y)
    ax.set(title='Time Series')
    lim = 1.1 * max(-dy.min(), dy.max())
    lim = -lim, lim
    rax.plot(dy)
    rax.set(ylim=lim, title='Residuals')
    sns.distplot(dy, bins=np.linspace(lim[0], lim[1], 22),
                 hist=True, kde=True, rug=True, ax=hax)
    hax.set(title='Residual Distribution')
    sm.graphics.tsa.plot_acf(dy, lags=acf_lags, ax=acfax)
    sm.graphics.tsa.plot_pacf(dy, lags=acf_lags, ax=pacfax)
    for a in axs.ravel():
        a.grid()
    plt.tight_layout()
    return fig, axs


def RMSE(Y, y):
    """Root-mean-square error."""
    return np.sqrt(np.mean((Y-y)**2))
def MAE(Y, y):
    """Mean absolute error."""
    return np.mean(np.abs(Y-y))
def MAPE(Y, y):
    """Mean absolute percent error."""
    return 100 * np.mean(np.abs((Y-y)/Y))
def MASE(Y, y):
    """TODO"""
    return np.nan # TODO


def tsaccuracy(Ytest, models):
    """Gather some metrics for a few models."""
    fs = RMSE, MAE, MAPE, MASE
    return pd.DataFrame({
        label: [ f(Ytest, model.predict(Ytest.index.min(), Ytest.index.max()))
                for f in (RMSE, MAE, MAPE, MASE) ]
        for (label, model) in models.items()
    }, index=[f.__name__ for f in fs]).T

def ciclean(ci_df):
    """Clean up conf_int() result column names."""
    ci_df = ci_df.copy()
    ci_df.columns = 'lower', 'upper'
    return ci_df


# legend_right = dict(loc='center left', bbox_to_anchor=[1, .5])


def generate_custom_ARMA(p, q, size = 1000, random_state = None):

    rng = np.random.default_rng(random_state)
    
    if (p == None):
        p = rng.integers(low = 0, high=3, size=1)
    if (q == None):
        q = rng.integers(low = 0, high=3, size=1)

    phi = []
    theta = []  
    rts_p = []
    rts_q = []

    AR_Ok = False 
    MA_Ok = False


    while((AR_Ok == False) or( MA_Ok == False)):
        # p, q = rng.integers(low = 0, high=3, size=2)
        # p, q = 0, 1
        # print(p, q, "\n","-----"*5)
        # if(((p + q) > 2) and ):
        #     p, q = rng.integers(low = 0, high=3, size=1)

        # create AR coefficients and enforce stationarity
        if p == 1:
            phi = (-np.random.uniform(low=-0.85, high=0.85, size=1)).tolist()
        elif p == 2:
            phi = (-np.random.uniform(low=-0.85, high=0.85, size=2)).tolist()
            # if ((phi[0] + phi[1] > 1) or (phi[1] - phi[0] > 1)):
            #     # regularity condition
            #     p = 0
            #     q = 0
            #     break
        else:
            phi = []

        ar_cff=[1] + phi
        rts_p = np.roots(np.flip(ar_cff))
        if((p > 1) and (np.absolute(rts_p).min() < 1)):
            continue
       
        AR_Ok = True
        # print("AR_OK")
        # print(ar_cff, rts_p)
        
        # create MA coefficients and enforce invertibility
        if q == 1:
            theta = np.random.uniform(low=-1, high=1, size=1).tolist()
        elif q == 2:
            theta = np.random.uniform(low=-1, high=1, size=2).tolist()
            # if ((theta[0] + theta[1] < -1) or (theta[0] - theta[1] > 1)):
            #     # regularity condition
            #     p = 0
            #     q = 0
            #     break
        else:
            theta = []
                
        ma_cff=[1] + theta
        rts_q = np.roots(np.flip(ma_cff))
        if((q > 1) and (np.absolute(rts_q).min() < 1)):
            continue
       
        MA_Ok =True
        # print("MA_OK")
        # print(ma_cff, rts_q)
    
    Y = ArmaProcess(ar_cff, ma_cff).generate_sample(nsample=size)
    Y_df = pd.DataFrame({"Y":Y})    
    
    return(Y_df, p, q, ar_cff, ma_cff, rts_p, rts_q)  
    
    
    
def plot_acf_pacf(df_ts, var, title="Time Series", lags = 10, plot_points=False):

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=False, sharey=False)

    ax0 = axs[0]
    df_ts[var].plot(color="blue", ax=ax0)
    if plot_points:
	    pd.DataFrame({"t":range(df_ts.shape[0]), 
	    	var:df_ts[var]}).plot(color="blue", kind="scatter", x="t", y=var, ax=ax0)
    ax0.set_title(title)
    ax0.grid(visible=True, which='both', axis='x')
    ax0.grid(visible=False, which='Major', axis='y')

    ax1 = axs[1]
    sm.graphics.tsa.plot_acf(df_ts[var].dropna(), ax=ax1, lags=lags, zero=False, title='ACF')
    ax1.set(ylim=(-1,1), xlabel='Lag', ylabel='acf')


    ax2 = axs[2]
    sm.graphics.tsa.plot_pacf(df_ts[var].dropna(), ax=ax2, lags=lags, zero=False, title='PACF')
    ax2.set(ylim=(-1,1), xlabel='Lag', ylabel='pacf')

    plt.tight_layout()
    plt.show();plt.close()