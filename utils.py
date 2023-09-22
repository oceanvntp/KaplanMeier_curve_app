import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from itertools import combinations
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st 
import sys
from io import BytesIO
import base64


# スタイル
style_list = ['solid', 'dashed', 'dashdot', 'dotted']
lancet_cp = ['#00468BFF', '#ED0000FF', '#42B540FF', '#0099B4FF',
             '#925E9FFF', '#FDAF91FF', '#AD002AFF', '#ADB6B6FF']
nejm_cp = ['#BC3C29FF', '#0072B5FF', '#E18727FF', '#20854EFF', 
           '#7876B1FF', '#6F99ADFF', '#FFDC91FF', '#EE4C97FF']



def generate_grayscale(x, white_value=0.8): #一番薄い色を変更するときはここ
    if x <= 0:
        return []

    step = white_value / (x - 1)  # x個の等間隔な数値を生成するためのステップ
    result = [round(i * step, 3) for i in range(x)]
    result = [str(val) for val in result]  # 四捨五入した後に文字列に変換
    return result

# ----------------------------------------
# カプランマイヤー曲線表示関数



def draw_km(df:pd.DataFrame, color:str or list='gray', size=(8, 4), by_subgroup:bool=True, 
            title:str='Kaplan Meier Curve', xlabel:str='生存日数', ylabel='生存率', 
            censor:bool=True, ci:bool=False, at_risk:bool=True, event_flag=1):
    
    subgroup = list(set(df.subgroup))
    fig, ax = plt.subplots(figsize=size, dpi=300)
    plt.suptitle(title)
    
    kmfs = [] # at_riskを正しく表示するため、fitしたインスタンスをリストに格納する
    if (len(subgroup) > 1) and by_subgroup: 
        for i, group in enumerate(subgroup):
            df_ = df[df.subgroup==group]
            kmf = KaplanMeierFitter()
            event_observed = df_.event.values
            if event_flag == 0:
                event_observed = 1 - event_observed
            kmf.fit(durations=df_.duration, event_observed=event_observed, label=group)
            if color == 'gray':  
                kmf.plot(show_censors=censor, ci_show=ci, color=color, 
                         linestyle=style_list[i], censor_styles={"marker": "|", "ms": 6, "mew": 0.75}) # matplotlibのマーカーと同じ。ms:長さ、mew:太さ
            else:
                kmf.plot(show_censors=censor, ci_show=ci, 
                         color=color[i], censor_styles={"marker": "|", "ms": 6, "mew": 0.75})
            kmfs.append(kmf)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if at_risk:
            add_at_risk_counts(*kmfs, rows_to_show=['At risk'])  # * でリストの中身を展開

        fig.tight_layout()            
        return fig
                
    
    else:
        kmf = KaplanMeierFitter()
        event_observed = df.event.values
        if event_flag == 0:
            event_observed = 1 - event_observed
        kmf.fit(durations=df.duration, event_observed=event_observed)
        if color == 'gray': 
            kmf.plot(show_censors=censor, ci_show=ci, color=color, 
                     label='_nolegend_', censor_styles={"marker": "|", "ms": 6, "mew": 0.75})
        else:
            kmf.plot(show_censors=censor, ci_show=ci, color=color[0], 
                     label='_nolegend_', censor_styles={"marker": "|", "ms": 6, "mew": 0.75})
        
        plt.gca.legend_ = None
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  
        if at_risk:
            add_at_risk_counts(kmf, rows_to_show=['At risk'])
   
        fig.tight_layout()
        return fig


#-----------------------------------
# 生存期間中央値、ci
def median_duration(df, event_flag=1):
    subgroup = list(set(df.subgroup))
    names, medians, cis_low, cis_high = [], [], [], []
    for group in subgroup:
        df_ = df[df.subgroup == group]
        event_observed = df_.event.values
        if event_flag == 0:
            event_observed = 1 - event_observed
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df_.duration, event_observed=event_observed)
        mst = kmf.median_survival_time_
        median_ci = median_survival_times(kmf.confidence_interval_)
        ci_low = median_ci.iloc[0, 0]
        ci_high = median_ci.iloc[0, 1]
        
        names.append(group)
        medians.append(mst)
        cis_low.append(ci_low)
        cis_high.append(ci_high)
    
    df_survival = pd.DataFrame({
        'subgroup':names,
        'median survival time':medians,
        '95% CI(lower)':cis_low,
        '95% CI(upper)':cis_high
    })
    return df_survival

#-----------------------------------
# Logrank検定
def logrank_p_table(df, event_flag=1):
    subgroup = list(set(df.subgroup))
    subgroup_combi = list(combinations(subgroup, 2))

    logrank_ps = []
    wilcoxon_ps = []
    names = []
    for combi in subgroup_combi:
        c1 = df[df['subgroup']==combi[0]]
        c2 = df[df['subgroup']==combi[1]]
        
        event_observed_c1 = c1.event.values
        event_observed_c2 = c2.event.values
        if event_flag == 0:
            event_observed_c1 = 1 - event_observed_c1
            event_observed_c2 = 1 - event_observed_c2
        logrank = logrank_test(c1.duration, c2.duration, event_observed_c1, event_observed_c2)
        wilcoxon = logrank_test(c1.duration, c2.duration, event_observed_c1, event_observed_c2, 
                                weightings='wilcoxon')
        logrank_p = logrank.p_value
        wilcoxon_p = wilcoxon.p_value
        logrank_ps.append(logrank_p)
        wilcoxon_ps.append(wilcoxon_p)
        names.append(combi[0]+'/'+combi[1])
    p_df = pd.DataFrame({'subgroup':names, 'logrank-p':logrank_ps, 'wilcoxon-p':wilcoxon_ps})
    return p_df

# p<0.05のとき色付け
def heighlight_value(val):
    if val < 0.05:
        return 'background-color: lightcoral'
    else:
        return ''

#-----------------------------------
# ハザード比

def hazard_table(df, inverse=False, event_flag=1):
    subgroup = list(set(df.subgroup))
    subgroup_combi = list(combinations(subgroup, 2))
    
    names = []
    hrs = []
    cis_low = []
    cis_high = []
    
    for combi in subgroup_combi:
        df_forcox = df[df['subgroup'].apply(lambda x: x in combi)]
        df_forcox['sub_label'] = df_forcox['subgroup'].apply(lambda x: combi.index(x) if x in combi else -1)
        df_forcox['event_0'] = df_forcox['event'].apply(lambda x: 1-x)
    
        
        cph = CoxPHFitter()
        df_forcox = df_forcox.drop('subgroup', axis=1)
        if event_flag == 1:
            df_forcox = df_forcox.drop('event_0', axis=1)
            cph = cph.fit(df_forcox, 'duration', 'event')
            
        elif event_flag == 0:
            df_forcox = df_forcox.drop('event', axis=1)
            cph = cph.fit(df_forcox, 'duration', 'event_0')
    
        hr = cph.hazard_ratios_.item()
        
        ci_low = np.exp(cph.confidence_intervals_.iloc[0,0])
        ci_high = np.exp(cph.confidence_intervals_.iloc[0,1])
        name = combi[1]+'/'+combi[0]
        
        if inverse:
            name = combi[0]+'/'+combi[1]
            hr = 1 / hr
            ci_low_ = 1 / ci_high
            ci_high_ = 1 / ci_low
            ci_low = ci_low_
            ci_high = ci_high_
            
        names.append(name)
        hrs.append(hr)
        cis_low.append(ci_low)
        cis_high.append(ci_high)
    df_cox = pd.DataFrame({'subgroup':names, 
                           'HR':hrs, 
                            '95% CI(lower)':cis_low,
                            '95% CI(upper)':cis_high})
    return df_cox

#-----------------------------------
#　画像ダウンロード

def download_button(fig, filename):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}.png">Download link: {filename} PNG(300 dpi)</a>'
    return href


######
#Wilcoxon検定