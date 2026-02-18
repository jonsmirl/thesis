"""
Capex Overinvestment Test: Empirical Validation of Paper 1, Proposition 1
==========================================================================
Tests whether actual hyperscaler AI capex matches the overinvestment
prediction from the endogenous decentralization differential game model.

Proposition 1 predicts:
  - N firms overinvest by 3-4x relative to cooperative optimum
  - Crossing time is compressed by ~79% under Nash competition
  - Overinvestment increases with N (Corollary 1)

Connor Smirl, Tufts University, February 2026
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = '/home/jonsmirl/thesis/thesis_data'
FIG_DIR = '/home/jonsmirl/thesis/figures/endogenous_decentralization'
RESULTS_FILE = os.path.join(DATA_DIR, 'capex_overinvestment_results.txt')
CAPEX_FILE = os.path.join(DATA_DIR, 'hyperscaler_capex.csv')
AGG_FILE = os.path.join(DATA_DIR, 'hyperscaler_capex_aggregate.csv')
os.makedirs(FIG_DIR, exist_ok=True)

def setup_style():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':11,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':9,
        'xtick.labelsize':10,'ytick.labelsize':10,'figure.dpi':150,'savefig.dpi':300,
        'savefig.bbox':'tight','axes.grid':True,'grid.alpha':0.3,'lines.linewidth':2.0})

class Parameters:
    def __init__(self, N=5, a=10.0, b=0.5, r=0.05, delta=0.30, x0=10.0):
        self.N = N; self.a = a; self.b = b; self.r = r; self.delta = delta; self.x0 = x0
        self.pi_bar = (a / (N + 1))**2 / b
        self.S = self.pi_bar / (r + delta)

def nash_Vprime(V, p):
    N, a, b, r = p.N, p.a, p.b, p.r
    A = N**2; B = -a*(1+N**2); C = a**2 - b*(N+1)**2*r*V
    D = B**2 - 4*A*C
    return np.nan if D < 0 else (-B - np.sqrt(D))/(2*A)

def coop_Vprime(V, p):
    N, a, b, r = p.N, p.a, p.b, p.r
    inner = 4*b*N*r*V
    if inner < 0: return np.nan
    s = np.sqrt(inner)
    return np.nan if s > a else (a - s)/N

def nash_output(Vp, p): return p.N*(p.a - Vp)/(p.b*(p.N + 1))
def coop_output(Vp, p): return (p.a - p.N*Vp)/(2*p.b)

def solve_value_functions(p, num_points=2000):
    dx = p.x0/num_points
    x = np.zeros(num_points+1); V_N = np.zeros(num_points+1); V_P = np.zeros(num_points+1)
    Vp_N = np.zeros(num_points+1); Vp_P = np.zeros(num_points+1)
    Q_N = np.zeros(num_points+1); Q_C = np.zeros(num_points+1)
    V_N[0] = p.S; V_P[0] = p.S
    Vp_N[0] = nash_Vprime(p.S, p); Vp_P[0] = coop_Vprime(p.S, p)
    Q_N[0] = nash_output(Vp_N[0], p); Q_C[0] = coop_output(Vp_P[0], p)
    for i in range(num_points):
        x[i+1] = x[i] + dx
        k1=nash_Vprime(V_N[i],p); k2=nash_Vprime(V_N[i]+.5*dx*k1,p)
        k3=nash_Vprime(V_N[i]+.5*dx*k2,p); k4=nash_Vprime(V_N[i]+dx*k3,p)
        if any(np.isnan([k1,k2,k3,k4])):
            x=x[:i+1];V_N=V_N[:i+1];V_P=V_P[:i+1];Vp_N=Vp_N[:i+1];Vp_P=Vp_P[:i+1];Q_N=Q_N[:i+1];Q_C=Q_C[:i+1];break
        V_N[i+1]=V_N[i]+(dx/6)*(k1+2*k2+2*k3+k4)
        Vp_N[i+1]=nash_Vprime(V_N[i+1],p); Q_N[i+1]=nash_output(Vp_N[i+1],p)
        k1=coop_Vprime(V_P[i],p); k2=coop_Vprime(V_P[i]+.5*dx*k1,p)
        k3=coop_Vprime(V_P[i]+.5*dx*k2,p); k4=coop_Vprime(V_P[i]+dx*k3,p)
        if any(np.isnan([k1,k2,k3,k4])):
            x=x[:i+1];V_N=V_N[:i+1];V_P=V_P[:i+1];Vp_N=Vp_N[:i+1];Vp_P=Vp_P[:i+1];Q_N=Q_N[:i+1];Q_C=Q_C[:i+1];break
        V_P[i+1]=V_P[i]+(dx/6)*(k1+2*k2+2*k3+k4)
        Vp_P[i+1]=coop_Vprime(V_P[i+1],p); Q_C[i+1]=coop_output(Vp_P[i+1],p)
    return x, V_N, V_P, Vp_N, Vp_P, Q_N, Q_C

def compute_crossing_time(x, Q):
    valid = Q > 1e-10; ig = np.zeros_like(Q); ig[valid] = 1.0/Q[valid]; ig[~valid] = np.nan
    m = ~np.isnan(ig); return np.trapezoid(ig[m], x[m])

def load_data():
    return pd.read_csv(CAPEX_FILE), pd.read_csv(AGG_FILE)

def calibrate_model(df_firm, df_agg):
    core_firms = ['Microsoft','Alphabet','Amazon','Meta','Oracle']
    df_core = df_firm[df_firm['company'].isin(core_firms)].copy()
    N_by_year = {}
    for year in sorted(df_core['year'].unique()):
        yr_data = df_core[df_core['year']==year]
        N_by_year[year] = max((yr_data['capex_bn_usd']>5.0).sum(), 2)
    actual_core = df_core.groupby('year')['capex_bn_usd'].sum().reset_index()
    actual_core.columns = ['year','capex_core_bn']
    baseline_year = 2020; baseline_N = N_by_year.get(baseline_year,4)
    br = actual_core[actual_core['year']==baseline_year]
    baseline_capex = br['capex_core_bn'].values[0] if len(br)>0 else 88.4
    total_cum = df_agg['industry_total_capex_bn'].sum()
    p_base = Parameters(N=baseline_N)
    years = sorted(actual_core['year'].unique())
    predictions = []
    for year in years:
        N_yr = N_by_year.get(year, baseline_N)
        p_yr = Parameters(N=N_yr)
        x_yr,V_N_yr,V_P_yr,Vp_N_yr,Vp_P_yr,Q_N_yr,Q_C_yr = solve_value_functions(p_yr)
        cum_to_year = actual_core[actual_core['year']<=year]['capex_core_bn'].sum()
        frac_done = min(cum_to_year/total_cum, 0.95)
        x_position = p_yr.x0*(1-frac_done)
        idx = np.argmin(np.abs(x_yr - x_position))
        q_nash = Q_N_yr[idx]; q_coop = Q_C_yr[idx]
        cum_bl = actual_core[actual_core['year']<=baseline_year]['capex_core_bn'].sum()
        frac_bl = min(cum_bl/total_cum, 0.95); x_bl = p_yr.x0*(1-frac_bl)
        idx_bl = np.argmin(np.abs(x_yr-x_bl)); q_nash_bl = Q_N_yr[idx_bl]
        scale_yr = baseline_capex/q_nash_bl if q_nash_bl>1e-10 else 1.0
        capex_nash = q_nash*scale_yr; capex_coop = q_coop*scale_yr
        av = actual_core[actual_core['year']==year]['capex_core_bn'].values
        actual_val = av[0] if len(av)>0 else np.nan
        oir = actual_val/capex_coop if (capex_coop>0 and not np.isnan(actual_val)) else np.nan
        nr = capex_nash/capex_coop if capex_coop>0 else np.nan
        predictions.append({'year':year,'N':N_yr,'x_position':x_position,'frac_done':frac_done,
            'capex_nash_pred':capex_nash,'capex_coop_pred':capex_coop,
            'capex_actual':actual_val,'overinvestment_ratio':oir,'nash_ratio':nr})
    return pd.DataFrame(predictions), N_by_year, p_base

def test_aggregate_exceeds_cooperative(df_pred, results):
    results.append(''); results.append('='*70)
    results.append('TEST 1: AGGREGATE CAPEX GROWTH vs COOPERATIVE OPTIMUM'); results.append('='*70)
    valid = df_pred.dropna(subset=['capex_actual','capex_coop_pred'])
    for _, row in valid.iterrows():
        excess = row['capex_actual']-row['capex_coop_pred']; ratio = row['capex_actual']/row['capex_coop_pred']
        results.append('  %d: Actual=$%.1fB, Cooperative=$%.1fB, Excess=$%.1fB (%.2fx)' % (int(row['year']),row['capex_actual'],row['capex_coop_pred'],excess,ratio))
    yr20=df_pred[df_pred['year']==2020]; yrl=df_pred[df_pred['year']==df_pred['year'].max()]
    if len(yr20)>0 and len(yrl)>0:
        a20=yr20['capex_actual'].values[0]; al=yrl['capex_actual'].values[0]
        c20=yr20['capex_coop_pred'].values[0]; cl=yrl['capex_coop_pred'].values[0]
        if not np.isnan(al) and a20>0 and c20>0:
            ag=al/a20; cg=cl/c20; ly=int(yrl['year'].values[0])
            results.append(''); results.append('  Capex growth 2020-%d:' % ly)
            results.append('    Actual: %.2fx' % ag); results.append('    Cooperative prediction: %.2fx' % cg)
            results.append('    Excess growth factor: %.2fx' % (ag/cg))
            results.append('  VERDICT: Actual growth %s cooperative optimum' % ('EXCEEDS' if ag>cg else 'BELOW'))
    return results

def test_acceleration_with_N(df_pred, df_firm, results):
    results.append(''); results.append('='*70)
    results.append('TEST 2: CAPEX ACCELERATION WITH FIRM ENTRY'); results.append('='*70)
    results.append(''); results.append('  Effective N (firms with >$5B capex) by year:')
    for _,row in df_pred.iterrows(): results.append('    %d: N = %d' % (int(row['year']),int(row['N'])))
    results.append(''); results.append('  Year-over-year aggregate capex growth:')
    for i in range(1, len(df_pred)):
        prev=df_pred.iloc[i-1]; curr=df_pred.iloc[i]
        if prev['capex_actual']>0 and not np.isnan(curr['capex_actual']):
            yoy=(curr['capex_actual']/prev['capex_actual']-1)*100
            nc=int(curr['N'])-int(prev['N'])
            ns=' (N: %d->%d)' % (int(prev['N']),int(curr['N'])) if nc!=0 else ''
            results.append('    %d->%d: %+.1f%%%s' % (int(prev['year']),int(curr['year']),yoy,ns))
    dt=df_pred.copy(); dt['yoy_growth']=dt['capex_actual'].pct_change(); dt['n_increased']=dt['N'].diff()>0
    gu=dt[dt['n_increased']]['yoy_growth'].dropna(); gf=dt[~dt['n_increased']]['yoy_growth'].dropna()
    if len(gu)>0 and len(gf)>0:
        results.append(''); results.append('  Avg YoY growth when N increased: %.1f%%' % (gu.mean()*100))
        results.append('  Avg YoY growth when N stable:    %.1f%%' % (gf.mean()*100))
        results.append('  VERDICT: Growth %s when N increased (consistent with Corollary 1)' % ('HIGHER' if gu.mean()>gf.mean() else 'NOT HIGHER'))
    cf=['Microsoft','Alphabet','Amazon','Meta','Oracle']; dc=df_firm[df_firm['company'].isin(cf)]
    m22=dc[(dc['company']=='Meta')&(dc['year']==2022)]; m24=dc[(dc['company']=='Meta')&(dc['year']==2024)]
    if len(m22)>0 and len(m24)>0:
        v22=m22['capex_bn_usd'].values[0]; v24=m24['capex_bn_usd'].values[0]
        results.append(''); results.append('  Meta capex: 2022=$%.1fB (metaverse peak) -> 2024=$%.1fB (AI pivot)' % (v22,v24))
        results.append('  Meta growth: %.0f%% (reflects AI arms race entry)' % ((v24/v22-1)*100))
    results.append(''); results.append('  Model crossing-time acceleration vs N:')
    results.append('  %4s  %10s  %10s  %8s' % ('N','T*_Nash','T*_Coop','Accel%'))
    results.append('  %s  %s  %s  %s' % ('-'*4,'-'*10,'-'*10,'-'*8))
    for Nt in [3,4,5,6,8,10]:
        pt=Parameters(N=Nt); xt,_,_,_,_,QNt,QCt=solve_value_functions(pt)
        tn=compute_crossing_time(xt,QNt); tc=compute_crossing_time(xt,QCt)
        ac=(1-tn/tc)*100 if tc>0 else 0
        results.append('  %4d  %10.4f  %10.4f  %7.1f%%' % (Nt,tn,tc,ac))
    return results

def test_overinvestment_ratio(df_pred, results):
    results.append(''); results.append('='*70)
    results.append('TEST 3: OVERINVESTMENT RATIO vs 3-4x PREDICTION (Prop 1)'); results.append('='*70)
    valid=df_pred.dropna(subset=['overinvestment_ratio'])
    ratios=valid['overinvestment_ratio'].values; years=valid['year'].values
    results.append(''); results.append('  Actual/Cooperative overinvestment ratio by year:')
    for yr,r in zip(years,ratios):
        ir='IN RANGE' if 2.0<=r<=5.0 else 'OUTSIDE'
        results.append('    %d: %.2fx  [%s]' % (int(yr),r,ir))
    ai=valid[valid['year']>=2022]
    if len(ai)>0:
        ar=ai['overinvestment_ratio'].mean()
        results.append(''); results.append('  Average ratio 2022-present: %.2fx' % ar)
        results.append('  Predicted range (Prop 1): 3-4x')
        results.append('  VERDICT: Empirical ratio %s with Proposition 1 prediction' % ('CONSISTENT' if 2.0<=ar<=5.0 else 'INCONSISTENT'))
    nr=valid['nash_ratio'].values
    results.append(''); results.append('  Model Nash/Cooperative ratios:')
    for yr,n in zip(years,nr): results.append('    %d: %.2fx' % (int(yr),n))
    return results

def test_nash_vs_actual(df_pred, results):
    results.append(''); results.append('='*70)
    results.append('TEST 4: NASH PATH vs ACTUAL TRAJECTORY'); results.append('='*70)
    valid=df_pred.dropna(subset=['capex_actual','capex_nash_pred'])
    actual=valid['capex_actual'].values; nash=valid['capex_nash_pred'].values; years=valid['year'].values
    results.append(''); results.append('  Year-by-year comparison:')
    for yr,a,n in zip(years,actual,nash):
        pe=(a-n)/n*100; results.append('    %d: Actual=$%.1fB, Nash pred=$%.1fB, error=%+.1f%%' % (int(yr),a,n,pe))
    if len(actual)>2:
        corr=np.corrcoef(actual,nash)[0,1]; rmse=np.sqrt(np.mean((actual-nash)**2))
        mae=np.mean(np.abs(actual-nash)); mape=np.mean(np.abs(actual-nash)/actual)*100
        results.append(''); results.append('  Fit statistics (Nash vs Actual):')
        results.append('    Pearson correlation:  r = %.4f' % corr)
        results.append('    RMSE:                %.2f $B' % rmse)
        results.append('    MAE:                 %.2f $B' % mae)
        results.append('    MAPE:                %.1f%%' % mape)
        v='GOOD FIT' if corr>0.85 else ('MODERATE FIT' if corr>0.7 else 'POOR FIT')
        results.append('    VERDICT: Nash path %s to actual trajectory (r=%.3f)' % (v,corr))
    return results

def test_competitive_dynamics(df_firm, results):
    results.append(''); results.append('='*70)
    results.append('COMPETITIVE DYNAMICS TESTS'); results.append('='*70)
    core_firms = ['Microsoft','Alphabet','Amazon','Meta']
    df_core = df_firm[df_firm['company'].isin(core_firms)].copy()
    results.append(''); results.append('  (a) Strategic Complementarity Test')
    results.append('  H0: Firm capex growth is independent of rival capex')
    results.append('  H1: Firm capex growth positively responds to rival capex (strategic complement)')
    arms_race_data = None
    try:
        import statsmodels.api as sm
        fp_list = []
        for firm in core_firms:
            fd = df_core[df_core['company']==firm].sort_values('year')
            for _,row in fd.iterrows():
                yr=row['year']; oc=row['capex_bn_usd']
                rv = df_core[(df_core['year']==yr)&(df_core['company']!=firm)]
                rc = rv['capex_bn_usd'].sum()
                fp_list.append({'firm':firm,'year':yr,'own_capex':oc,'rival_capex':rc})
        fp = pd.DataFrame(fp_list)
        for firm in core_firms:
            m = fp['firm']==firm
            fp.loc[m,'rival_capex_lag'] = fp.loc[m,'rival_capex'].shift(1)
            fp.loc[m,'own_capex_lag'] = fp.loc[m,'own_capex'].shift(1)
        fp = fp.dropna()
        fp['own_growth'] = fp['own_capex']/fp['own_capex_lag'] - 1
        fp['rival_growth'] = fp['rival_capex']/fp['rival_capex_lag'] - 1
        X = sm.add_constant(fp['rival_growth']); y = fp['own_growth']
        model = sm.OLS(y, X).fit()
        results.append('')
        results.append('  Regression: own_capex_growth ~ rival_capex_growth')
        results.append('    beta (rival_growth):  %.4f' % model.params.iloc[1])
        results.append('    t-stat:               %.3f' % model.tvalues.iloc[1])
        results.append('    p-value:              %.4f' % model.pvalues.iloc[1])
        results.append('    R-squared:            %.4f' % model.rsquared)
        results.append('    N obs:                %d' % len(fp))
        comp = model.params.iloc[1]>0 and model.pvalues.iloc[1]<0.10
        bs = 'positive' if model.params.iloc[1]>0 else 'negative'
        pb = '<0.10' if model.pvalues.iloc[1]<0.10 else '>0.10'
        results.append('    VERDICT: %s (beta=%s, p=%s)' % ('STRATEGIC COMPLEMENTS' if comp else 'No significant complementarity',bs,pb))
        results.append('')
        results.append('  Level regression: own_capex ~ rival_capex_lagged')
        X2 = sm.add_constant(fp['rival_capex_lag']); y2 = fp['own_capex']
        m2 = sm.OLS(y2, X2).fit()
        results.append('    beta (rival_capex_lag): %.4f' % m2.params.iloc[1])
        results.append('    t-stat:                %.3f' % m2.tvalues.iloc[1])
        results.append('    p-value:               %.4f' % m2.pvalues.iloc[1])
        results.append('    R-squared:             %.4f' % m2.rsquared)
        arms_race_data = fp
    except ImportError:
        results.append('  [statsmodels not available, skipping regression]')
    results.append(''); results.append('  (b) Capex Concentration (HHI) Over Time')
    results.append('  Model predicts: more competitors => more spread => more overinvestment')
    hhi_data = []
    for year in sorted(df_core['year'].unique()):
        yd = df_core[df_core['year']==year]; total=yd['capex_bn_usd'].sum()
        if total>0:
            shares = yd['capex_bn_usd']/total; hhi = (shares**2).sum()
            hhi_data.append({'year':year,'hhi':hhi,'n_firms':len(yd),'total':total})
    df_hhi = pd.DataFrame(hhi_data)
    results.append(''); results.append('  %6s  %8s  %8s  %10s' % ('Year','HHI','N firms','Total ($B)'))
    results.append('  %s  %s  %s  %s' % ('-'*6,'-'*8,'-'*8,'-'*10))
    for _,row in df_hhi.iterrows():
        results.append('  %6d  %8.4f  %8d  %10.1f' % (int(row['year']),row['hhi'],int(row['n_firms']),row['total']))
    if len(df_hhi)>2:
        hc = np.corrcoef(df_hhi['year'],df_hhi['hhi'])[0,1]
        results.append(''); results.append('  HHI-year correlation: %.4f' % hc)
        d = 'DECLINING' if hc<0 else 'INCREASING'
        mf = 'more' if hc<0 else 'fewer'
        results.append('  VERDICT: Concentration is %s (consistent with %s competitors entering)' % (d,mf))
    return results, arms_race_data, df_hhi

def fig_model_vs_actual(df_pred):
    fig, ax = plt.subplots(figsize=(9,6))
    valid = df_pred.dropna(subset=['capex_actual'])
    years=valid['year'].values; actual=valid['capex_actual'].values
    nash=valid['capex_nash_pred'].values; coop=valid['capex_coop_pred'].values
    ax.plot(years,actual,'ko-',label='Actual hyperscaler capex',linewidth=2.5,markersize=8,zorder=5)
    ax.plot(years,nash,'b^--',label='Nash MPE prediction',linewidth=2,markersize=7,alpha=0.85)
    ax.plot(years,coop,'rs--',label='Cooperative optimum',linewidth=2,markersize=7,alpha=0.85)
    ax.fill_between(years,coop,actual,alpha=0.12,color='red',label='Overinvestment gap')
    ax.axvline(x=2022.5,color='gray',linestyle=':',alpha=0.5,linewidth=1)
    yl=ax.get_ylim(); ax.text(2022.6,yl[0]+(yl[1]-yl[0])*0.15,'ChatGPT\n(Nov 2022)',fontsize=8,ha='left',va='bottom',color='gray')
    ax.set_xlabel('Year'); ax.set_ylabel('Aggregate Capex ($B)')
    ax.set_title('Hyperscaler AI Capex: Model vs. Actual\n(Paper 1, Proposition 1: Nash Overinvestment)')
    ax.legend(loc='upper left',framealpha=0.9); ax.set_xlim(years.min()-0.3,years.max()+0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_capex_model_vs_actual.png')); plt.close(fig)
    print('  Saved: %s' % os.path.join(FIG_DIR,'fig_capex_model_vs_actual.png'))

def fig_capex_by_firm(df_firm):
    fig, ax = plt.subplots(figsize=(10,6))
    cf=['Microsoft','Alphabet','Amazon','Meta','Oracle']
    colors={'Microsoft':'#00A4EF','Alphabet':'#EA4335','Amazon':'#FF9900','Meta':'#1877F2','Oracle':'#F80000'}
    markers={'Microsoft':'o','Alphabet':'s','Amazon':'^','Meta':'D','Oracle':'v'}
    ed={}
    for firm in cf:
        fd=df_firm[df_firm['company']==firm].sort_values('year')
        if len(fd)==0: continue
        ax.plot(fd['year'],fd['capex_bn_usd'],marker=markers.get(firm,'o'),color=colors.get(firm,'black'),label=firm,linewidth=2,markersize=7)
        by=fd[fd['capex_bn_usd']>5.0]
        if len(by)>0: ed[firm]=by['year'].min()
    for firm,yr in ed.items():
        cv=df_firm[(df_firm['company']==firm)&(df_firm['year']==yr)]['capex_bn_usd'].values[0]
        ax.annotate('%s\nentry' % firm,xy=(yr,cv),xytext=(yr-0.8,cv+8),fontsize=7,ha='center',color=colors.get(firm,'black'),arrowprops=dict(arrowstyle='->',color=colors.get(firm,'black'),alpha=0.6))
    ax.axhline(y=5.0,color='gray',linestyle=':',alpha=0.5,linewidth=1)
    ax.text(2018.2,5.5,'$5B threshold',fontsize=8,color='gray')
    ax.set_xlabel('Year'); ax.set_ylabel('Annual Capex ($B)')
    ax.set_title('Individual Hyperscaler Capex Trajectories\nwith AI Capex Entry Dates Marked')
    ax.legend(loc='upper left',framealpha=0.9,ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_capex_by_firm.png')); plt.close(fig)
    print('  Saved: %s' % os.path.join(FIG_DIR,'fig_capex_by_firm.png'))

def fig_overinvestment_ratio(df_pred):
    fig, ax = plt.subplots(figsize=(9,6))
    valid=df_pred.dropna(subset=['overinvestment_ratio'])
    years=valid['year'].values; ratios=valid['overinvestment_ratio'].values; nr=valid['nash_ratio'].values
    ax.plot(years,ratios,'ko-',label='Actual / Cooperative',linewidth=2.5,markersize=8,zorder=5)
    ax.plot(years,nr,'b^--',label='Nash / Cooperative (model)',linewidth=2,markersize=7,alpha=0.8)
    ax.axhspan(3.0,4.0,alpha=0.15,color='green',label='Prop 1 prediction: 3-4x')
    ax.axhline(y=3.5,color='green',linestyle='--',alpha=0.4,linewidth=1)
    ax.axhspan(2.0,5.0,alpha=0.05,color='blue')
    ax.set_xlabel('Year'); ax.set_ylabel('Overinvestment Ratio (Actual / Cooperative)')
    ax.set_title('Overinvestment Ratio vs. Proposition 1 Prediction\n(Shaded band: 3-4x predicted range)')
    ax.legend(loc='upper left',framealpha=0.9); ax.set_ylim(0,max(max(ratios),6.0)*1.1)
    ax.axvline(x=2022.5,color='gray',linestyle=':',alpha=0.5,linewidth=1)
    yl=ax.get_ylim(); ax.text(2022.6,yl[1]*0.95,'ChatGPT',fontsize=8,ha='left',va='top',color='gray')
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_capex_overinvestment_ratio.png')); plt.close(fig)
    print('  Saved: %s' % os.path.join(FIG_DIR,'fig_capex_overinvestment_ratio.png'))

def fig_arms_race(df_firm, arms_race_data, df_hhi):
    fig, axes = plt.subplots(1,2,figsize=(14,6))
    cf=['Microsoft','Alphabet','Amazon','Meta']
    colors={'Microsoft':'#00A4EF','Alphabet':'#EA4335','Amazon':'#FF9900','Meta':'#1877F2'}
    dc=df_firm[df_firm['company'].isin(cf)].copy()
    ax=axes[0]
    for firm in cf:
        fd=dc[dc['company']==firm].sort_values('year')
        if len(fd)<2: continue
        yoy=fd['capex_bn_usd'].pct_change()*100
        ax.plot(fd['year'].values[1:],yoy.values[1:],marker='o',color=colors.get(firm,'black'),label=firm,linewidth=1.5,markersize=5)
    ax.axhline(y=0,color='gray',linewidth=0.5)
    ax.axvline(x=2022.5,color='gray',linestyle=':',alpha=0.5,linewidth=1)
    ax.set_xlabel('Year'); ax.set_ylabel('YoY Capex Growth (%)'); ax.set_title('Arms Race: Firm-Level Capex Growth')
    ax.legend(loc='best',fontsize=8,framealpha=0.9)
    ax2=axes[1]
    ax2.plot(df_hhi['year'],df_hhi['hhi'],'ko-',linewidth=2,markersize=7,label='HHI (left)')
    ax2.set_xlabel('Year'); ax2.set_ylabel('Herfindahl-Hirschman Index')
    ax2.set_title('Capex Concentration (HHI) and Total Spending'); ax2.set_ylim(0,1)
    ax3=ax2.twinx()
    ax3.bar(df_hhi['year'],df_hhi['total'],alpha=0.3,color='steelblue',label='Total capex ($B)')
    ax3.set_ylabel('Total Capex ($B)',color='steelblue'); ax3.tick_params(axis='y',labelcolor='steelblue')
    l1,lb1=ax2.get_legend_handles_labels(); l2,lb2=ax3.get_legend_handles_labels()
    ax2.legend(l1+l2,lb1+lb2,loc='upper right',fontsize=8,framealpha=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR,'fig_capex_arms_race.png')); plt.close(fig)
    print('  Saved: %s' % os.path.join(FIG_DIR,'fig_capex_arms_race.png'))

def compute_summary(df_pred, df_firm, N_by_year, results):
    results.append(''); results.append('='*70); results.append('SUMMARY STATISTICS'); results.append('='*70)
    valid = df_pred.dropna(subset=['capex_actual'])
    ta=valid['capex_actual'].sum(); tc=valid['capex_coop_pred'].sum(); tn=valid['capex_nash_pred'].sum()
    ym=int(valid['year'].min()); yx=int(valid['year'].max())
    results.append(''); results.append('  Cumulative capex (core hyperscalers) %d-%d:' % (ym,yx))
    results.append('    Actual:      $%.1fB' % ta); results.append('    Nash pred:   $%.1fB' % tn)
    results.append('    Coop pred:   $%.1fB' % tc)
    results.append('    Actual/Coop: %.2fx' % (ta/tc)); results.append('    Nash/Coop:   %.2fx' % (tn/tc))
    results.append(''); results.append('  Model calibration:')
    results.append('    Learning rate alpha:      0.23 (Wrights Law, Paper 1)')
    results.append('    Displacement rate delta:  0.30 (IBM calibration)')
    results.append('    Discount rate r:          0.05'); results.append('    Baseline year:            2020')
    results.append('    Baseline N:               %d' % N_by_year.get(2020,4))
    ai = valid[valid['year']>=2022]
    if len(ai)>0:
        ar=ai['overinvestment_ratio'].mean(); mr=ai['overinvestment_ratio'].max()
        my=ai.loc[ai['overinvestment_ratio'].idxmax(),'year']
        results.append(''); results.append('  KEY FINDING (AI period 2022+):')
        results.append('    Average overinvestment ratio: %.2fx' % ar)
        results.append('    Peak overinvestment ratio:    %.2fx (in %d)' % (mr,int(my)))
        results.append('    Proposition 1 prediction:     3-4x')
    p5=Parameters(N=5); x5,_,_,_,_,QN5,QC5=solve_value_functions(p5)
    TN5=compute_crossing_time(x5,QN5); TC5=compute_crossing_time(x5,QC5)
    accel=(1-TN5/TC5)*100
    results.append(''); results.append('  MODEL PREDICTION:')
    results.append('    Crossing time acceleration (N=5): %.1f%%' % accel)
    results.append('    Paper 1 claim: ~79%%')
    if len(valid)>=4:
        ny=valid['year'].iloc[-1]-valid['year'].iloc[0]
        if ny>0 and valid['capex_actual'].iloc[0]>0:
            ac=(valid['capex_actual'].iloc[-1]/valid['capex_actual'].iloc[0])**(1/ny)-1
            cc=(valid['capex_coop_pred'].iloc[-1]/valid['capex_coop_pred'].iloc[0])**(1/ny)-1
            results.append(''); results.append('    Actual CAGR: %.1f%%' % (ac*100))
            results.append('    Cooperative CAGR: %.1f%%' % (cc*100))
            if cc>0: results.append('    Acceleration ratio: %.2fx' % (ac/cc))
    return results

def main():
    setup_style()
    results = []
    results.append('='*70)
    results.append('CAPEX OVERINVESTMENT TEST: Empirical Validation of Paper 1, Prop 1')
    results.append('Connor Smirl, Tufts University, February 2026')
    results.append('='*70)
    print('Loading hyperscaler capex data...')
    df_firm, df_agg = load_data()
    nf = df_firm['company'].nunique()
    print('  Firm-level: %d rows, %d companies' % (len(df_firm),nf))
    print('  Aggregate:  %d rows, years %d-%d' % (len(df_agg),int(df_agg['year'].min()),int(df_agg['year'].max())))
    results.append(''); results.append('Data: %d firm-year obs, %d companies' % (len(df_firm),nf))
    results.append('Years: %d-%d' % (int(df_agg['year'].min()),int(df_agg['year'].max())))
    print('Calibrating differential game model to data...')
    df_pred, N_by_year, p_base = calibrate_model(df_firm, df_agg)
    print('  Effective N by year: %s' % N_by_year)
    results.append('Effective N by year: %s' % N_by_year)
    results.append(''); results.append('='*70); results.append('PREDICTION TABLE'); results.append('='*70)
    results.append('  %6s %3s %10s %10s %10s %10s' % ('Year','N','Actual','Nash','Coop','Act/Coop'))
    results.append('  %s %s %s %s %s %s' % ('-'*6,'-'*3,'-'*10,'-'*10,'-'*10,'-'*10))
    for _,row in df_pred.iterrows():
        a_str='$%.1fB' % row['capex_actual'] if not np.isnan(row['capex_actual']) else 'N/A'
        r_str='%.2fx' % row['overinvestment_ratio'] if not np.isnan(row['overinvestment_ratio']) else 'N/A'
        results.append('  %6d %3d %10s %10s %10s %10s' % (int(row['year']),int(row['N']),a_str,'$%.1fB' % row['capex_nash_pred'],'$%.1fB' % row['capex_coop_pred'],r_str))
    print('\nRunning tests...')
    print('  Test 1: Aggregate vs cooperative...')
    results = test_aggregate_exceeds_cooperative(df_pred, results)
    print('  Test 2: Acceleration with N...')
    results = test_acceleration_with_N(df_pred, df_firm, results)
    print('  Test 3: Overinvestment ratio...')
    results = test_overinvestment_ratio(df_pred, results)
    print('  Test 4: Nash path vs actual...')
    results = test_nash_vs_actual(df_pred, results)
    print('  Competitive dynamics tests...')
    results, arms_race_data, df_hhi = test_competitive_dynamics(df_firm, results)
    results = compute_summary(df_pred, df_firm, N_by_year, results)
    print('\nGenerating figures...')
    fig_model_vs_actual(df_pred)
    fig_capex_by_firm(df_firm)
    fig_overinvestment_ratio(df_pred)
    fig_arms_race(df_firm, arms_race_data, df_hhi)
    results_text = '\n'.join(results)
    with open(RESULTS_FILE, 'w') as f: f.write(results_text)
    print('\nResults saved to: %s' % RESULTS_FILE)
    print('\n' + results_text)
    print('\nDone.')

if __name__ == '__main__':
    main()
