#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 00:21:28 2018

@author: Zhehao Li
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from sklearn import linear_model
from pandas.tseries.offsets import Day, MonthEnd
from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


AShare_indicator = pd.read_pickle('AShareEODDerivativeIndicator.pkl')#归母净利润，净资产，日收盘价
Stock_industry = pd.read_csv('行业对应关系.csv')
capital_expenditure = pd.read_excel('财务数据.xlsx',sheet_name='资本开支')
operation_revenue = pd.read_excel('财务数据.xlsx',sheet_name='营业收入')
research_expenditure = pd.read_excel('财务数据.xlsx',sheet_name='研发费用')


###############################################################################

###############################################################################
'''1. PB与ROE'''


def IndustryFinance(AShare_data):
    #行业市值(自然日)
    stock_value = pd.pivot_table(AShare_data,values='S_VAL_MV',columns='S_INFO_WINDCODE',index='TRADE_DT')                  
    stock_value['SUM'] = stock_value.sum(axis=1)
    
    #行业净资产(自然日)
    stock_assets = pd.pivot_table(AShare_data,values='NET_ASSETS_TODAY',columns='S_INFO_WINDCODE',index='TRADE_DT')
    stock_assets['SUM'] = stock_assets.sum(axis=1)
    
    #A股净利润（TTM）
    stock_profit = pd.pivot_table(AShare_data,values='NET_PROFIT_PARENT_COMP_TTM',columns='S_INFO_WINDCODE',index='TRADE_DT')
    
    #A股未来12个月净利润
    stock_future_profit = stock_profit.shift(periods=-365)
    stock_future_profit = stock_future_profit.dropna(how='all',axis=0)
    #行业中A股净利润的和
    stock_future_profit['SUM'] = stock_future_profit.sum(axis=1)
    return {'市值':stock_value,'净资产':stock_assets,'未来净利润':stock_future_profit}


def Compute_ROE_PB(industry_data):
    stock_value = industry_data['市值']
    stock_assets = industry_data['净资产']
    stock_future_profit = industry_data['未来净利润']
    
    #行业ROE(未来12个月净利润/当下净资产)
    stock_ROE = stock_future_profit['SUM']/stock_assets['SUM']
    stock_ROE.index = pd.DatetimeIndex(stock_ROE.index)
    stock_ROE = stock_ROE.dropna()
    
    #行业PB
    stock_PB = stock_value['SUM']/stock_assets['SUM']
    stock_PB.index = pd.DatetimeIndex(stock_PB.index)
    stock_PB = stock_PB.reset_index()
    stock_PB.columns = ['TRADE_DT','PB']
    stock_PB = pd.merge(stock_ROE.reset_index(),stock_PB.reset_index(),on='TRADE_DT',how='left')
    stock_PB = stock_PB.drop(['SUM','index'],axis=1)
    stock_PB = stock_PB.set_index('TRADE_DT')
    return stock_ROE,stock_PB

###############################################################################

def Compute_Relative_PB_ROE(AShare_data,Market_ROE,Market_PB,name):
    #按行业分类提取数据
    AShare_industry = AShare_data[AShare_data['INDUSTRY'] == name]
    #计算行业市值，净资产，未来12个月滚动利润
    industry_data = IndustryFinance(AShare_industry)
    #计算行业ROE，PB
    stock_ROE,stock_PB = Compute_ROE_PB(industry_data)
    #对齐日期
    ROE = pd.merge(Market_ROE,stock_ROE.reset_index(),on='TRADE_DT',how='left')
    PB = pd.merge(stock_PB.reset_index(),Market_PB,on='TRADE_DT',how='left')
    ROE = ROE.dropna()
    #合并表格，对齐时间
    PB_ROE = pd.merge(ROE,PB,on='TRADE_DT',how='left')
    return PB_ROE


###############################################################################


#计算决定系数
def computeR2(y,y_hat):
    y = y.reshape(-1,1)
    y_hat = y_hat.reshape(-1,1)
    y_bar = y.mean()
    return 1 - np.sum((y-y_hat)**2) / np.sum((y-y_bar)**2)


#建立线性回归魔仙
def LinearRegression(x,y):
    assert len(x) == len(y)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    regr = linear_model.LinearRegression()
    regr.fit(x,y)
    a = regr.coef_
    b = regr.intercept_
    predict_y = regr.predict(x)
    R2 = computeR2(y,predict_y)
    par_dict = {'coef':a,'intercept':b,'predict_value':predict_y,'R2':R2}
    return par_dict 
    

def Plot_PB_ROE(X,Y,Par,name):
    a = Par['coef']
    b = Par['intercept']
    R2 = Par['R2']
    Predict_Y = Par['predict_value']
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(X,Y,s=3,color='orange',label='y=' + '%.3f'%a + 'x+' + '%.2f'%b)
    ax.plot(X,Predict_Y,color='blue',label='R^2 = '+'%.3f'%R2)
    ax.set_title(name)
    ax.set_xlabel('相对ROE')
    ax.set_ylabel('相对PB')
    ax.legend(loc='upper left')
    plt.savefig(name+'相对PB_ROE.jpg',dpi=400)

###############################################################################

#绘制所有行业 相对PB和相对ROE关系图
def AShare_Basis_Main(AShare_data,Market_ROE,Market_PB,Industry_name):
    for name in Industry_name:
        PB_ROE = Compute_Relative_PB_ROE(AShare_data,Market_ROE,Market_PB,name)
        #相对ROE
        X = PB_ROE['SUM']/PB_ROE['ROE_TTM']
        X = X.values
        #相对PB
        Y = PB_ROE['PB']/PB_ROE['PB_LF']
        Y = Y.values
        Par = LinearRegression(X,Y)
        Plot_PB_ROE(X,Y,Par,name)

 
'''存在问题及改进:

1. 数据库导出的净资产，净利润均为日数据，而由于每个上市公司财报公布日期不同，因此无法将日数据与理论财报公布日(季度数据)对齐，所以最后计算得到行业
净利润，行业净资产以及ROE，PB均为日数据，无法与理论财报日期对齐；

2. 市场ROE（上证指数ROE）数据日期与理论财报日期相同（暂时不知道是如何对齐理论财报日的）,尽管后期将季度数据扩展为月度数据，由于日期不同，这就造成了计算行业相对ROE和相对PB
可能存在些许误差（但估计误差小于10%），导致拟合结果欠佳；


改进：
1. 我们选取的行业分类为一级行业，因此单个行业内股票很多，导致计算行业的总资产、总利润后，拟合效果欠佳，若选择二级行业分类，拟合效果会提升；
若进一步选择行业内大盘股拟合效果会比较好（长江策略做法）

2. 个股净利润、净资产使用Wind提供的季度数据(理论财报日数据)将显著提升拟合效果

'''



#初始化数据
AShare_Basis = pd.merge(AShare_indicator,Stock_industry,on='S_INFO_WINDCODE')
AShare_Basis['S_VAL_MV'] = AShare_Basis['S_VAL_PB_NEW']*AShare_Basis['NET_ASSETS_TODAY']

industry_name = list(Stock_industry['INDUSTRY'].drop_duplicates().values[:-1])

#导入上证指数PB和rOE
Market_PB = pd.read_pickle('SZZZ_PB.pkl')
Market_PB['TRADE_DT'] = pd.DatetimeIndex(Market_PB['TRADE_DT'])
#Market_PB = Market_PB.set_index('TRADE_DT')

Market_ROE = pd.read_pickle('SZZZ_ROE.pkl')
Market_ROE['REPORT_PERIOD'] = pd.DatetimeIndex(Market_ROE['REPORT_PERIOD'])
Market_ROE = Market_ROE.sort_values('REPORT_PERIOD')

#将市场ROE季度数据扩展成月度数据
new_DF = []
for i in range(Market_ROE.shape[0]):
    year = str(Market_ROE['REPORT_PERIOD'][i])[:4]
    data = Market_ROE['ROE_TTM'][i]
    tempDf = pd.DataFrame({'TRADE_DT':pd.date_range(start=year+'0101',periods=3,freq='m'),'ROE_TTM':data})
    new_DF.append(tempDf)

new_Market_ROE = pd.concat(new_DF,ignore_index=True)


#运行
AShare_Basis_Main(AShare_Basis,new_Market_ROE,Market_PB,industry_name)



###############################################################################

###############################################################################


'''2. 资本开支与营业收入变化时间序列图'''

#合并资本开支，营业收入，研发费用三项数据
AShare_Finance = pd.merge(capital_expenditure,operation_revenue,on=['股票代码','财报日期'])
AShare_Finance = AShare_Finance.merge(research_expenditure,on=['股票代码','财报日期'])

#合并行业分类数据
AShare_Finance.columns = ['S_INFO_WINDCODE','REPORT_DT','CAP_EXPENDITURE','OP_REVENUE','RES_EXPENDITURE']
AShare_Finance = AShare_Finance.merge(Stock_industry,on='S_INFO_WINDCODE')
AShare_Finance['REPORT_DT'] = pd.to_datetime(AShare_Finance['REPORT_DT'],format='%Y%m%d')


industry_name = list(Stock_industry['INDUSTRY'].drop_duplicates().values[:-1])


#计算三年年均复合增长率
def AvgIncreasePerYear(data_series):
    #计算单季度平均复合增长率
    temp = data_series/data_series.shift(periods=4) #
    temp = temp**(1/4)
    #累计三年复合增长率(3*4季度)
    temp_series = pd.Series(index=data_series.index)  
    for d in range(12,temp.shape[0]):
        temp_series.values[d] = temp[d-12:d].prod()
    #三年年均复合增长率
    cagr = temp_series**(1/3)-1
    return cagr


#生成所有行业年均复合增长率表格
def IndustryCAGR(AShare_data,industry,values_name):
    industry_cagr = pd.DataFrame()
    for name in industry:
        #计算各行业数据总和
        data = pd.pivot_table(AShare_Finance[AShare_Finance['INDUSTRY'] == name],values=values_name,index='REPORT_DT',columns='S_INFO_WINDCODE')
        data = data[data.index>pd.to_datetime('2002-12-31')] #截取2003年开始的数据
        data['SUM'] = data.sum(axis=1)
        #计算年平均复合增长率
        industry_cagr[name] = AvgIncreasePerYear(data['SUM'])
    return industry_cagr



#生成所有行业年均相对复合增长率(时间截面上各行业增速归一化)
def RelativeCAGR(AShare_data,industry,values_name):
    industry_rel_cagr = pd.DataFrame()
    industry_cagr = IndustryCAGR(AShare_data,industry,values_name)
    MAX = industry_cagr.max(axis=1)
    MIN = industry_cagr.min(axis=1)
    for name in industry:
        industry_rel_cagr[name] = (industry_cagr[name] - MIN)/(MAX-MIN)
    return industry_rel_cagr
  

#绘图，营业收入和资本支出关系
def PlotCAGR(X,Y,fig_name):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(111) 
    ax.plot(X,Y,marker='o',ms=4,label=fig_name)
    ax.set_xlim(-0.1,1.1)
    ax.set_ylim(-0.1,1.1)
    for i in range(len(X)):
        if i%4 == 0 and not pd.isna(X[i]):
            date = str(X.index[i])[:4]
            plt.text(X[i],Y[i]+0.02,date, ha='center', va= 'bottom',fontsize=8)    
    ax.legend(loc='upper right')
    ax.set_xlabel('营业收入增速')
    ax.set_ylabel('资本开支增速')
    #plt.show()
    plt.savefig(fig_name+'资本开支_营业收入.jpg',dpi=400)


def AShare_Finance_Main(AShare_data,industry):    
    #资本支出
    CAPITAL = RelativeCAGR(AShare_data,industry,values_name='CAP_EXPENDITURE')
    #营业收入
    REVENUE = RelativeCAGR(AShare_data,industry,values_name='OP_REVENUE')
    #研发费用
   # RESEARCH = RelativeCAGR(AShare_data,industry,values_name='RES_EXPENDITURE')
    
    for name in industry:
        x = REVENUE[name]
        y = CAPITAL[name]
        PlotCAGR(x,y,name)

#运行
AShare_Finance_Main(AShare_data=AShare_Finance,industry=industry_name)



