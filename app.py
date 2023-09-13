import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from itertools import combinations
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st 

##################################
# カプランマイヤー曲線表示関数
style_list = ['solid', 'dashed', 'dashdot', 'dotted']
grayscale = ['0.', '0.35', '0.7', '0.9']
lancet_cp = ['#00468BFF', '#ED0000FF', '#42B540FF', '#0099B4FF']
nejm_cp = ['#BC3C29FF', '#0072B5FF', '#E18727FF', '#20854EFF']

def draw_km(df:pd.DataFrame, color:str or list='gray', size=(8, 4), by_subgroup:bool=True, 
            title:str='Kaplan Meier Curve', xlabel:str='生存日数', ylabel='生存率', 
            censor:bool=True, ci:bool=False, at_risk:bool=True):
    
    subgroup = list(set(df.subgroup))
    fig, ax = plt.subplots(figsize=size, dpi=300)
    plt.suptitle(title)
    
    kmfs = []
    if (len(subgroup) > 0) and by_subgroup: 
        for i, group in enumerate(subgroup):
            df_ = df[df.subgroup==group]
            kmf = KaplanMeierFitter()
            kmf.fit(durations=df_.duration, event_observed=df_.event, label=group)
            if color == 'gray':  
                kmf.plot(show_censors=censor, ci_show=ci, color=color, linestyle=style_list[i])
            else:
                kmf.plot(show_censors=censor, ci_show=ci, color=color[i])
            kmfs.append(kmf)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if at_risk:
            add_at_risk_counts(*kmfs, rows_to_show=['At risk'])  # * でリストの中身を展開

        fig.tight_layout()            
        return fig
                
    
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(durations=df.duration, event_observed=df.event)
        if color == 'gray': 
            kmf.plot(show_censors=censor, ci_show=ci, color=color, label='_nolegend_')
        else:
            kmf.plot(show_censors=censor, ci_show=ci, color=color[0], label='_nolegend_')
        
        plt.gca.legend_ = None
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)  
        if at_risk:
            add_at_risk_counts(kmf, rows_to_show=['At risk'])
   
        fig.tight_layout()
        return fig

#-----------------------------------
# Logrank検定
def logrank_p_table(df):
    subgroup = list(set(df.subgroup))
    subgroup_combi = list(combinations(subgroup, 2))

    ps = []
    names = []
    for combi in subgroup_combi:
        c1 = df[df['subgroup']==combi[0]]
        c2 = df[df['subgroup']==combi[1]]
        logrank = logrank_test(c1.duration, c2.duration, c1.event, c2.event)
        p = logrank.p_value
        ps.append(p)
        names.append(combi[0]+'/'+combi[1])
    p_df = pd.DataFrame({'subgroup':names, 'p-value':ps})
    return p_df

#-----------------------------------
# ハザード比
def hazard_table(df, inverse=False):
    subgroup = list(set(df.subgroup))
    subgroup_combi = list(combinations(subgroup, 2))
    
    names = []
    hrs = []
    cis_low = []
    cis_high = []
    
    for combi in subgroup_combi:
        df_forcox = df[df['subgroup'].apply(lambda x: x in combi)]
        df_forcox['sub_label'] = df_forcox['subgroup'].apply(lambda x: combi.index(x) if x in combi else -1)
        cph = CoxPHFitter()
        df_forcox = df_forcox.drop('subgroup', axis=1)
        cph = cph.fit(df_forcox, 'duration', 'event')
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
    df_cox = pd.DataFrame({'subgroup':names, 'HR':hrs, 
                        '95% CI(lower)':cis_low,'95% CI(upper)':cis_high})
    return df_cox

    
##################################
#　本体
st.title('カプランマイヤー曲線作成App')

# Excelファイルのアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx", "xls"])
st.text('列名をduration, event, subgroup(任意)としたexcelファイルをアップロードしてください。')
st.text('duration: イベントまでの期間 day, month, yearsいずれも可。')
st.text('event: 観察期間中のイベントの有無。イベント発生が1、イベント未発生は0。')
st.text('subgroup: 群間比較をしたいときはここにラベルを入れてください。')
st.write("テンプレートExcel [link](https://github.com/oceanvntp/KaplanMeier_curve_app/raw/main/sample_table/%E3%83%86%E3%83%B3%E3%83%97%E3%83%AC%E3%83%BC%E3%83%88.xlsx)")

st.write('---')
title = st.text_input('グラフタイトル',value='Kaplan Meier Curve')
col1, col2 = st.columns(2)
with col1:
    xlabel = st.text_input('横軸のラベル', value='期間')
with col2:
    ylabel = st.text_input('縦軸のラベル', value='生存率')

##################################
# サイドバー
style = st.sidebar.selectbox('スタイル', ('グレースケール', 'グレー', 'NEJM', 'Lancet'))
if style == 'グレースケール':
    color = grayscale
elif style == 'グレー':
    color = 'gray'
elif style == 'NEJM':
    color = nejm_cp
elif style == 'Lancet':
    color = lancet_cp
    
st.sidebar.write('---')
st.sidebar.text('図表サイズ')
size_x = st.sidebar.slider('横', min_value=5, max_value=10, value=8)
size_y = st.sidebar.slider('縦', min_value=5, max_value=10, value=6)
size = (size_x, size_y)

st.sidebar.write('---')
by_sub = st.sidebar.selectbox('グループ', ('グループごと', '全体集団'))
if by_sub == 'グループごと':
    by_subgroup = True 
elif by_sub == '全体集団':
    by_subgroup = False
    
st.sidebar.write('---')
censor_flag = st.sidebar.selectbox('打ち切り表示', ('有', '無'))
if censor_flag=='有':
    censor = True
elif censor_flag=='無':
    censor = False 

st.sidebar.write('---')
ci_flag = st.sidebar.selectbox('信頼区間表示', ('無', '有'))
if ci_flag=='無':
    ci = False
elif ci_flag=='有':
    ci = True
    
st.sidebar.write('---')
at_risk_ = st.sidebar.selectbox('N at risk表示', ('有', '無'))
at_risk = True if at_risk_=='有' else False


##################################
# ファイルアップロード後の処理
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, header=0)
    fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                  title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, ci=ci, at_risk=at_risk)
    st.pyplot(fig)
    st.text('Logrank検定')
    p_df = logrank_p_table(df)

    st.table(p_df)
    st.text('ハザード比(対象群/参照群)')
    inverse = st.checkbox('対象, 参照反転')
    cox_df = hazard_table(df, inverse=inverse)
    st.table(cox_df)


# ファイルが無いときはサンプルを表示できるように
elif (uploaded_file is None):
    st.write('---')
    st.text('下のボタンを押すとサンプルが表示されます。')
    sample = st.button('サンプル表示')
    if sample:
        df = pd.read_excel('sample_table/sampleExcel.xlsx', header=0)
        fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                    title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, ci=ci, at_risk=at_risk)
        st.pyplot(fig)
        st.text('Logrank検定')
        p_df = logrank_p_table(df)
        st.table(p_df)
        st.text('ハザード比(対照群/参照群)')
        inverse = st.checkbox('対象, 参照反転')
        cox_df = hazard_table(df, inverse=inverse)
        st.table(cox_df)
        

