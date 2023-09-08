import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st 


# カプランマイヤー曲線表示関数
style_list = ['solid', 'dashed', 'dashdot', 'dotted']
lancet_cp = ['#00468BFF', '#ED0000FF', '#42B540FF', '#0099B4FF']
nejm_cp = ['#BC3C29FF', '#0072B5FF', '#E18727FF', '#20854EFF']

def draw_km(df:pd.DataFrame, color:str or list='gray', size=(8, 4), by_subgroup:bool=True, 
            title:str='Kaplan Meier Curve', xlabel:str='生存日数', ylabel='生存率', 
            censor:bool=True, ci:bool=False):
    
    subgroup = list(set(df.subgroup))
    fig, ax = plt.subplots(figsize=size)
    plt.suptitle(title)
    
    if (len(subgroup) > 0) and by_subgroup: 
        for i, group in enumerate(subgroup):
            df_ = df[df.subgroup==group]
            kmf = KaplanMeierFitter()
            kmf.fit(durations=df_.duration, event_observed=df_.event, label=group)
            if color == 'gray':  
                kmf.plot(show_censors=censor, ci_show=ci, color=color, linestyle=style_list[i])
            else:
                kmf.plot(show_censors=censor, ci_show=ci, color=color[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)            
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
        plt.plot()   
        return fig

#　本体
st.title('カプランマイヤー曲線作成App')


# Excelファイルのアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx", "xls"])
st.text('列名をduration, event, subgroup(任意)としたexcelファイルをアップロードしてください。')
st.text('duration: イベントまでの期間 day, month, yearsいずれも可')
st.text('event: 観察期間中のイベントの有無。イベント発生が1、イベント未発生は0')
st.text('subgroup: 群間比較をしたいときはここにラベルを入れてください。')

st.write('---')
title = st.text_input('グラフタイトル',value='Kaplan Meier Curve')
col1, col2 = st.columns(2)
with col1:
    xlabel = st.text_input('横軸のラベル', value='期間')
with col2:
    ylabel = st.text_input('縦軸のラベル', value='生存率')


# サイドバー
style = st.sidebar.selectbox('スタイル', ('グレー', 'NEJM', 'Lancet'))
if style == 'グレー':
    color = 'gray'
elif style == 'NEJM':
    color = nejm_cp
elif style == 'Lancet':
    color = lancet_cp
    
st.sidebar.write('---')
st.sidebar.text('図表サイズ')
size_x = st.sidebar.slider('横', min_value=5, max_value=10, value=8)
size_y = st.sidebar.slider('縦', min_value=5, max_value=10, value=5)
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


# ファイルアップロード後の処理
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, header=0)
    fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                  title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, ci=ci)
    st.pyplot(fig)



elif (uploaded_file is None):
    st.write('---')
    st.text('下のボタンを押すとサンプルが表示されます。')
    sample = st.button('サンプル表示')
    if sample:
        df = pd.read_excel('sample_table/sampleExcel.xlsx', header=0)
        fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                    title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, ci=ci)
        st.pyplot(fig)
    

