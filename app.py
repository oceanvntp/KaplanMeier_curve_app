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
from utils import generate_grayscale, draw_km, median_duration, logrank_p_table, heighlight_value, hazard_table, download_button, custom_color_and_style




##################################
# スタイル
style_list = ['solid', 'dashed', 'dashdot', 'dotted']
# grayscale = ['0.', '0.3', '0.6', '0.8', '0.9']
lancet_cp = ['#00468BFF', '#ED0000FF', '#42B540FF', '#0099B4FF',
             '#925E9FFF', '#FDAF91FF', '#AD002AFF', '#ADB6B6FF']
nejm_cp = ['#BC3C29FF', '#0072B5FF', '#E18727FF', '#20854EFF', 
           '#7876B1FF', '#6F99ADFF', '#FFDC91FF', '#EE4C97FF']
##################################
##################################


#　本体
st.title('カプランマイヤー曲線作成App')

# Excelファイルのアップロード
uploaded_file = st.file_uploader("Excelファイルをアップロードしてください", type=["xlsx", "xls"])
st.text('列名をduration, event, subgroupとしたexcelファイルをアップロードしてください。')
st.text('※列名は必須です。')
st.write("テンプレートExcel [link](https://github.com/oceanvntp/KaplanMeier_curve_app/raw/main/sample_table/%E3%83%86%E3%83%B3%E3%83%97%E3%83%AC%E3%83%BC%E3%83%88.xlsx)")
st.write('  ')
st.text('duration: イベントまでの期間 day, month, yearsいずれも可。')
st.text('event: 観察期間中のイベントの有無(1 or 0)')
st.text('subgroup: 群間比較をしたいときはここにラベルを入れてください。')


st.write('---')
title = st.text_input('グラフタイトル',value='')
col1, col2 = st.columns(2)
with col1:
    xlabel = st.text_input('横軸のラベル', value='期間')
with col2:
    ylabel = st.text_input('縦軸のラベル', value='生存率')

##################################
##################################
##################################
# サイドバー
color_style = st.sidebar.selectbox('スタイル', ('グレースケール', 'グレー', 'NEJM', 'Lancet', 'カスタム'))
if color_style == 'グレースケール':
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=0)
        df = df.fillna({'subgroup':'None'})
        color = generate_grayscale(len(set(df.subgroup)))
    else:
        df = pd.read_excel('sample_table/sampleExcel.xlsx', header=0)
        df = df.fillna({'subgroup':'None'})
        color = generate_grayscale(len(set(df.subgroup)))
        
    linestyle_choice = False

        
elif color_style == 'グレー':
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, header=0)
        df = df.fillna({'subgroup':'None'})
        if len(set(df.subgroup)) > 4:
            st.sidebar.write('このスタイルは4群まで対応しています。')
        else:
            color = 'gray'
    linestyle_choice = False
            
elif color_style == 'NEJM':
    color = nejm_cp
    linestyle_choice = False
elif color_style == 'Lancet':
    color = lancet_cp
    linestyle_choice = False
elif color_style == 'カスタム':
    linestyle_choice = True

style_choice_list = None
    
#-----------------------------------
st.sidebar.write('---')
st.sidebar.text('図表サイズ')
size_x = st.sidebar.slider('横', min_value=5, max_value=12, value=8)
size_y = st.sidebar.slider('縦', min_value=5, max_value=12, value=6)
size = (size_x, size_y)

#-----------------------------------
st.sidebar.write('---')
event_flag = st.sidebar.selectbox('イベント発生', (1, 0))

#-----------------------------------
st.sidebar.write('---')
by_sub = st.sidebar.selectbox('グループ', ('グループごと', '全体集団'))
if by_sub == 'グループごと':
    by_subgroup = True 
elif by_sub == '全体集団':
    by_subgroup = False
    
#-----------------------------------
st.sidebar.write('---')
censor_flag = st.sidebar.selectbox('打ち切り表示', ('有', '無'))
if censor_flag=='有':
    censor = True
elif censor_flag=='無':
    censor = False 
    
#-----------------------------------
st.sidebar.write('---')
ci_flag = st.sidebar.selectbox('信頼区間表示', ('無', '有'))
ci = False if ci_flag=='無' else True

    
#-----------------------------------   
st.sidebar.write('---')
at_risk_ = st.sidebar.selectbox('N at risk表示', ('有', '無'))
at_risk = True if at_risk_=='有' else False

#-----------------------------------  
st.sidebar.write('---')
fontsize = st.sidebar.slider('N at risk サイズ', min_value=8, max_value=14, value=11)
#-----------------------------------   
st.sidebar.write('---')
fontname = st.sidebar.selectbox('N at risk　フォント', ('Arial', 'Times New Roman', 'Helvetica'))


##################################
# ファイルアップロード後の処理
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, header=0)
    df = df.dropna(subset=['duration', 'event'])
    df = df.fillna({'subgroup':'None'})
    subgroup = df.subgroup.unique()
    if color_style=='カスタム':
        color, linestyle = custom_color_and_style(subgroup)
        style_choice_list = linestyle
    fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                    linestyle_choice=linestyle_choice, style_choice_list=style_choice_list,
                title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, 
                ci=ci, at_risk=at_risk, event_flag=event_flag,
                fontsize=fontsize, fontname=fontname)
    st.pyplot(fig)
    # if st.button('ダウンロード'):
    st.markdown(download_button(fig, "km_curve"), unsafe_allow_html=True)
    
    st.text('●生存期間')
    st.table(median_duration(df, event_flag=event_flag))
    subgroup = df.subgroup.unique()
    if len(subgroup) >= 2:
        st.text('●Logrank/Wilcoxon検定')
        p_df = logrank_p_table(df, event_flag=event_flag)
        st.table(p_df.style.applymap(heighlight_value, subset=['logrank-p', 'wilcoxon-p']))
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
        subgroup = df.subgroup.unique()
        if color_style=='カスタム':
            color, linestyle = custom_color_and_style(subgroup)
            style_choice_list = linestyle
        fig = draw_km(df, color=color, size=size, by_subgroup=by_subgroup,
                      linestyle_choice=linestyle_choice, style_choice_list=style_choice_list,
                    title=title, xlabel=xlabel, ylabel=ylabel, censor=censor, 
                    ci=ci, at_risk=at_risk, event_flag=event_flag,
                    fontsize=fontsize, fontname=fontname)
        st.pyplot(fig)
        
        st.text('●生存期間')
        st.table(median_duration(df, event_flag=event_flag))
        st.text('●Logrank/Wilcoxon検定')
        p_df = logrank_p_table(df, event_flag=event_flag)
        st.table(p_df.style.applymap(heighlight_value, subset=['logrank-p', 'wilcoxon-p']))
        st.text('●ハザード比(対照群/参照群)')
        inverse = st.checkbox('対象, 参照反転')
        cox_df = hazard_table(df, inverse=inverse, event_flag=event_flag)
        st.table(cox_df)
        

st.write('---')
st.text('統計解析環境')
st.text(f'Python ver: {sys.version}')
st.text('Numpy ver: 1.25.2')
st.text('lifelines ver: 0.27.7')
