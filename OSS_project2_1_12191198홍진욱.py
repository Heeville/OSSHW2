#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from pandas import Series, DataFrame

basedata=pd.read_csv('2019_kbo_for_kaggle_v2.csv')

print('2-1번')
for year in range(2015, 2019):
    basedata_year = basedata[basedata['year'] == year]

    hits_top10 =basedata_year.sort_values('H',ascending=False).head(10)
    print(f"\n{year}년 안타(H) 상위 10명")
    print(hits_top10.loc[:,['batter_name','H','age','G','PA','AB','year']])
    
    avg_top10 =basedata_year.sort_values('avg',ascending=False).head(10)
    print(f"{year}년 타율(avg) 상위 10명")
    print(avg_top10.loc[:,['batter_name','avg','age','G','PA','AB','year']])
    
    homerun_top10 =basedata_year.sort_values('HR',ascending=False).head(10)
    print(f"{year}년 홈런(HR) 상위 10명")
    print(homerun_top10.loc[:,['batter_name','HR','age','G','PA','AB','year']])

    obp_top10 =basedata_year.sort_values('OBP',ascending=False).head(10)
    print(f"{year}년 출루율(OBP) 상위 10명")
    print(obp_top10.loc[:,['batter_name','OBP','age','G','PA','AB','year']])   
    
print('\n2-2번\n')    
basedata2018=basedata[basedata['year'] == 2018]

hposu=basedata2018[basedata2018['cp']=='포수'].sort_values('war',ascending=False).head(1)
h1B=basedata2018[basedata2018['cp']=='1루수'].sort_values('war',ascending=False).head(1)
h2B=basedata2018[basedata2018['cp']=='2루수'].sort_values('war',ascending=False).head(1)
h3B=basedata2018[basedata2018['cp']=='3루수'].sort_values('war',ascending=False).head(1)
hSS=basedata2018[basedata2018['cp']=='유격수'].sort_values('war',ascending=False).head(1)
hLF=basedata2018[basedata2018['cp']=='좌익수'].sort_values('war',ascending=False).head(1)
hCF=basedata2018[basedata2018['cp']=='중견수'].sort_values('war',ascending=False).head(1)
hRF=basedata2018[basedata2018['cp']=='우익수'].sort_values('war',ascending=False).head(1)

hightestcollection=pd.DataFrame(index=['포수','1루수','2루수','3루수','유격수','좌익수','중견수','우익수'],columns=['선수이름','war',])

print('각 포지션 별 2018년 war(승리기여도)가 가장 높은 선수 이름과 war 값 출력결과\n')
hightestcollection.loc['포수'] = [hposu.iloc[0][0], hposu.iloc[0][23]]
hightestcollection.loc['1루수'] = [h1B.iloc[0][0], h1B.iloc[0][23]]
hightestcollection.loc['2루수'] = [h2B.iloc[0][0], h2B.iloc[0][23]]
hightestcollection.loc['3루수'] = [h3B.iloc[0][0], h3B.iloc[0][23]]
hightestcollection.loc['유격수'] = [hSS.iloc[0][0], hSS.iloc[0][23]]
hightestcollection.loc['좌익수'] = [hLF.iloc[0][0], hLF.iloc[0][23]]
hightestcollection.loc['중견수'] = [hCF.iloc[0][0], hCF.iloc[0][23]]
hightestcollection.loc['우익수'] = [hRF.iloc[0][0], hRF.iloc[0][23]]

print(hightestcollection)
                                                                                                      
print('\n2-3번\n')  
salarycorr=dict()
salarycorr['득점(R)']=basedata['R'].corr(basedata['salary'])
salarycorr['안타(H)']=basedata['H'].corr(basedata['salary'])
salarycorr['홈런(HR)']=basedata['HR'].corr(basedata['salary'])
salarycorr['타점(RBI)']=basedata['RBI'].corr(basedata['salary'])
salarycorr['도루(SB)']=basedata['SB'].corr(basedata['salary'])
salarycorr['승리기여도(war)']=basedata['war'].corr(basedata['salary'])
salarycorr['타율(avg)']=basedata['avg'].corr(basedata['salary'])
salarycorr['출루율(OBP)']=basedata['OBP'].corr(basedata['salary'])
salarycorr['장타율(SLG)']=basedata['SLG'].corr(basedata['salary'])

highestcorr=-1

for key,value in salarycorr.items():
    print(f"{key}와/과 연봉의 상관계수: {value}")
    if(value>highestcorr):
        highestcorr=value
        field=key

print(f"\n연봉과 가장 높은 상관계수를 갖는 항목은 \'{field}\' 이고 이때 상관계수 값은 {highestcorr} 이다" )


# In[ ]:




