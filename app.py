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


##################################
##################################
##################################
# スタイル
style_list = ['solid', 'dashed', 'dashdot', 'dotted']
# grayscale = ['0.', '0.3', '0.6', '0.8', '0.9']
lancet_cp = ['#00468BFF', '#ED0000FF', '#42B540FF', '#0099B4FF',
             '#925E9FFF', '#FDAF91FF', '#AD002AFF', '#ADB6B6FF']
nejm_cp = ['#BC3C29FF', '#0072B5FF', '#E18727FF', '#20854EFF', 
           '#7876B1FF', '#6F99ADFF', '#FFDC91FF', '#EE4C97FF']

def generate_grayscale(x):
    if x <= 0:
        return []

    step = 0.8 / (x - 1)  # x個の等間隔な数値を生成するためのステップ
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

    ps = []
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
        p = logrank.p_value
        ps.append(p)
        names.append(combi[0]+'/'+combi[1])
    p_df = pd.DataFrame({'subgroup':names, 'p-value':ps})
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
    
##################################
##################################
##################################
##################################

#　本体
st.title('カプランマイヤー曲線作成App')

# Excelファイルのアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx", "xls"])
st.text('列名をduration, event, subgroupとしたexcelファイルをアップロードしてください。')
st.text('※列名は必須です。その他の列は削除してください。')
st.write("テンプレートExcel [link](https://github.com/oceanvntp/KaplanMeier_curve_app/raw/main/sample_table/%E3%83%86%E3%83%B3%E3%83%97%E3%83%AC%E3%83%BC%E3%83%88.xlsx)")
st.write('  ')
st.text('duration: イベントまでの期間 day, month, yearsいずれも可。')
st.text('event: 観察期間中のイベントの有無(1 or 0)')
st.text('subgroup: 群間比較をしたいときはここにラベルを入れてください。')
st.text('          ※現状ラベルがないとエラーが出ます。単群でも適当にラベルを入れてください。')


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
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=0)
        df = df.fillna({'subgroup':'None'})
        color = generate_grayscale(len(set(df.subgroup)))
    else:
        pass
elif style == 'グレー':
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=0)
        df = df.fillna({'subgroup':'None'})
        if len(set(df.subgroup)) > 4:
            st.sidebar.write('このスタイルは4群まで対応しています。')
        else:
            color = 'gray'
            
elif style == 'NEJM':
    color = nejm_cp
elif style == 'Lancet':
    color = lancet_cp
    
st.sidebar.write('---')
st.sidebar.text('図表サイズ')
size_x = st.sidebar.slider('横', min_value=5, max_value=12, value=8)
size_y = st.sidebar.slider('縦', min_value=5, max_value=12, value=6)
size = (size_x, size_y)

st.sidebar.write('---')
event_flag = st.sidebar.selectbox('イベント発生', (1, 0))

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
    df = df.fillna({'subgroup':'None'})
    fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                  title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, 
                  ci=ci, at_risk=at_risk, event_flag=event_flag)
    st.pyplot(fig)
    # if st.button('ダウンロード'):
    st.markdown(download_button(fig, "km_curve"), unsafe_allow_html=True)
    
    st.text('●生存期間')
    st.table(median_duration(df, event_flag=event_flag))
    subgroup = list(set(df.subgroup))
    if len(subgroup) >= 2:
        st.text('●Logrank検定')
        p_df = logrank_p_table(df, event_flag=event_flag)
        st.table(p_df.style.applymap(heighlight_value, subset=['p-value']))
        st.text('●ハザード比(対象群/参照群)')
        inverse = st.checkbox('対象, 参照反転')
        cox_df = hazard_table(df, inverse=inverse, event_flag=event_flag)
        st.table(cox_df)


# ファイルが無いときはサンプルを表示できるように
elif (uploaded_file is None):
    st.write('---')
    st.text('チェックするとサンプルが表示されます。')
    sample = st.checkbox('サンプル表示')
    if sample:
        df = pd.read_excel('sample_table/sampleExcel.xlsx', header=0)
        fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                    title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, 
                    ci=ci, at_risk=at_risk, event_flag=event_flag)
        st.pyplot(fig)
        
        st.text('●生存期間')
        st.table(median_duration(df, event_flag=event_flag))
        st.text('●Logrank検定')
        p_df = logrank_p_table(df, event_flag=event_flag)
        st.table(p_df.style.applymap(heighlight_value, subset=['p-value']))
        st.text('●ハザード比(対照群/参照群)')
        inverse = st.checkbox('対象, 参照反転')
        cox_df = hazard_table(df, inverse=inverse, event_flag=event_flag)
        st.table(cox_df)
        

st.write('---')
st.text('統計解析環境')
st.text(f'Python ver: {sys.version}')
st.text('Numpy ver: 1.25.2')
st.text('lifelines ver: 0.27.7')
