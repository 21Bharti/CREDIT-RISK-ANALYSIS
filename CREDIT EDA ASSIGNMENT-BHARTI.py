#!/usr/bin/env python
# coding: utf-8

# ## CREDIT EDA ASSIGNMENT

# #### Objectives
# This case study aims to identify patterns which indicate if a client has difficulty paying their instalments which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc. This will ensure that the consumers capable of repaying the loan are not rejected. Identification of such applicants using EDA is the aim of this case study.
# 
# In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilise this knowledge for its portfolio and risk assessment.
# 
# To develop your understanding of the domain, you are advised to independently research a little about risk analytics - understanding the types of variables and their significance should be enough.

# ### 1. Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# ### Analyzing 'application_data' datasets

# ### 2. Loading data 

# In[2]:


application_data= pd.read_csv('application_data.csv')


# In[3]:


application_data


# ### 3. Data Pre-processing

# * Checking rows & columns

# In[4]:


application_data.shape


# * Understanding the data with 'info' & 'describe' function

# In[5]:


application_data.info(verbose = True,null_counts = True)


# In[6]:


application_data.describe()


# ### 3(a). Handling missing values

# In[7]:


pd.set_option('display.max_columns',150)
pd.set_option('display.max_rows',150)
round(application_data.isnull().sum()/len(application_data.index)*100,2)


# * Droping columns for higher missing values (>40%), as imputing these large no missing values will bias the analysis.  

# In[8]:


app_data=application_data.dropna(thresh=0.6*len(application_data), axis=1)


# In[9]:


round(app_data.isnull().sum()/len(app_data.index)*100,2)


# * Handling missing values for 'OCCUPATION_TYPE', i.e. 31.35%

# In[10]:


app_data['OCCUPATION_TYPE'].value_counts(normalize=True)*100


# * As there are large no of applicant's whose occupation data is not provided, droping this important column or even imputing occupation may bias the analysis. So, I'll just give name 'Not known' for these missing data. 

# In[11]:


app_data['OCCUPATION_TYPE'].fillna('Not known', inplace = True)


# * Handling missing values for 'NAME_TYPE_SUITE', i.e. 0.420418%

# In[12]:


app_data['NAME_TYPE_SUITE'].value_counts()


# In[13]:


app_data['NAME_TYPE_SUITE'].mode()[0]


# * Missing data is not high, so I will replace missing data with 'Mode' value because 'NAME_TYPE_SUITE' is a categorical feature. 

# In[14]:


app_data['NAME_TYPE_SUITE']= app_data['NAME_TYPE_SUITE'].fillna(app_data['NAME_TYPE_SUITE'].mode()[0])


# In[15]:


app_data['NAME_TYPE_SUITE'].value_counts()


# * Handling missing value for 'AMT_ANNUITY'

# In[16]:


app_data['AMT_ANNUITY'].describe()


# In[17]:


sns.boxplot(app_data['AMT_ANNUITY'])
plt.show()
app_data['AMT_ANNUITY'].median()


# * Seeing box plot, we can find out that there are lots of outliers present in 'AMT_ANNUITY'. So, I'll impute it with median. 

# In[18]:


app_data['AMT_ANNUITY']=app_data['AMT_ANNUITY'].fillna(app_data['AMT_ANNUITY'].median())


# In[19]:


app_data[['AMT_GOODS_PRICE','CNT_FAM_MEMBERS']].head()


# * CNT_FAM_MEMBERS

# In[20]:


app_data['CNT_FAM_MEMBERS'].value_counts()


# In[21]:


app_data['CNT_FAM_MEMBERS'].describe()


# In[22]:


sns.boxplot(app_data['CNT_FAM_MEMBERS'])
plt.show()
app_data['CNT_FAM_MEMBERS'].median()


# * Box plot shows, it has outliers so, its better to impute missing values with 'Median' rather than 'Mode' & 'Mean'

# In[23]:


app_data['CNT_FAM_MEMBERS']=app_data['CNT_FAM_MEMBERS'].fillna(app_data['CNT_FAM_MEMBERS'].median())


# In[24]:


app_data.isnull().sum()/len(app_data.index)*100


# #### Analyzing missing value 13.5% for 6 columns
# * AMT_REQ_CREDIT_BUREAU_HOUR
# * AMT_REQ_CREDIT_BUREAU_DAY
# * AMT_REQ_CREDIT_BUREAU_WEEK
# * AMT_REQ_CREDIT_BUREAU_MON
# * AMT_REQ_CREDIT_BUREAU_QRT
# * AMT_REQ_CREDIT_BUREAU_YEAR

# In[25]:


app_data[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# In[26]:


app_data[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].mode()


# * Its better to not imputing the data as nullable value is large & may create bias in insights.

# #### Our analysis doesnt include these columns, so I will drop these columns.

# In[27]:


app_data.drop(['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','DAYS_LAST_PHONE_CHANGE'], inplace= True, axis=1)


# In[28]:


app_data.dropna(inplace=True)


# In[29]:


app_data.isnull().sum()/len(app_data.index)*100


# ### 3(b). Handling incorrect data types

# In[30]:


app_data.dtypes


# #### CNT_FAM_MEMBERS
# * CNT_FAM_MEMBERS datatypes given is float, Family members count will be integer always. So i will convert it to int. 

# In[31]:


app_data.CNT_FAM_MEMBERS= app_data.CNT_FAM_MEMBERS.astype(int)
app_data.CNT_FAM_MEMBERS.dtypes


# #### CODE_GENDER

# In[32]:


app_data['CODE_GENDER'].value_counts()


# * In CODE_GENDER, 'F' gender is having highest count, i.e. mode of CODE_GENDER is 'F'.

# In[33]:


app_data[app_data['CODE_GENDER']=='XNA']


# In[34]:


app_data['CODE_GENDER'].mode()[0]


# * There are 4 nos of data given, where CODE_GENDER is 'XNA', either data was not provided to bank person or any other reason might exist for this.
# * Will replace this 'XNA' with mode, i.e. 'F' for easier analysis. 

# In[35]:


app_data['CODE_GENDER']= app_data['CODE_GENDER'].replace('XNA','F')


# In[36]:


app_data['CODE_GENDER'].value_counts()


# #### 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'

# In[37]:


app_data['DAYS_BIRTH'].value_counts()


# #### We will observe that 'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH' are given in negative.
# * Might be the bank person has entered data in negative to indicate the days before. 
# * For standardizing & easier calculation, I have converted it into positive. 

# In[38]:


# DAYS_BIRTH are given in negative, so convert it to positive


# In[39]:


app_data['DAYS_BIRTH']=app_data['DAYS_BIRTH'].abs()


# In[40]:


app_data['YEARS_BIRTH']=app_data['DAYS_BIRTH'].apply(lambda x: round(float(x/365)))


# In[41]:


app_data['DAYS_EMPLOYED'].value_counts()


# In[42]:


# DAYS_EMPLOYED are given in negative, so convert it to positive


# In[43]:


app_data['DAYS_EMPLOYED']=app_data['DAYS_EMPLOYED'].abs()


# * We see that there are 44,056 data-counts which representing 'DAYS_EMPLOYED' as 1001 years, which is practically impossible.

# In[44]:


app_data['YEARS_EMPLOYED']= app_data['DAYS_EMPLOYED'].apply(lambda x: round(float(x/365)))


# In[45]:


app_data['DAYS_EMPLOYED'].value_counts()


# In[46]:


app_data['YEARS_EMPLOYED'].value_counts()


# In[47]:


app_data.head()


# In[48]:


app_data['DAYS_REGISTRATION'].value_counts()


# In[49]:


# DAYS_REGISTRATION are given in negative, so convert it to positive


# In[50]:


app_data['DAYS_REGISTRATION']=app_data['DAYS_REGISTRATION'].abs()


# In[51]:


app_data['DAYS_ID_PUBLISH'].value_counts()


# In[52]:


# DAYS_ID_PUBLISH are given in negative, so convert it to positive


# In[53]:


app_data['DAYS_ID_PUBLISH']=app_data['DAYS_ID_PUBLISH'].abs()


# In[54]:


app_data['DAYS_ID_PUBLISH'].value_counts()


# #### Checking 'AMT_INCOME_TOTAL' & 'AMT_CREDIT' attributes data.

# In[55]:


app_data[['AMT_INCOME_TOTAL','AMT_CREDIT']].describe()


# #### AMT_INCOME_TOTAL & AMT_CREDIT data given is continuous, and its difficult to do Analysing on this data. Let's categories them in ranges for better understandings.

# In[56]:


# Lets take variables 'Bin' & 'Ranges' for categorizing them

Bins=[0,100000,300000,500000,700000,900000,1100000,1300000,1500000,1700000,1900000,2100000,2300000,2500000,2700000,3000000,
      3200000,3500000,3800000,4000000,4300000,4500000,4700000,5000000]

Ranges= ['0-100000','100000-300000','300000-500000','500000-700000','700000-900000','900000-1100000','1100000-1300000',
         '1300000-1500000','1500000-1700000','1500000-1900000','1900000-2100000','2100000-2300000','2300000-2500000',
         '2500000-2700000','2700000-3000000','3000000-3200000','3200000-3500000','3500000-3800000','3800000-4000000',
         '4000000-4300000','4300000-4500000','4500000-4700000','4700000 and above']


# In[57]:


app_data['AMT_INCOME_RANGE']= pd.cut(app_data['AMT_INCOME_TOTAL'], Bins, labels=Ranges)
app_data['AMT_CREDIT_RANGE']= pd.cut(app_data['AMT_CREDIT'], Bins, labels=Ranges)


# In[58]:


app_data.columns


# #### I want to do analysis on these attributes so have made a new dataframe for these columns by using 'copy' function

# In[59]:


app_data_new = app_data[['TARGET','CODE_GENDER', 'CNT_CHILDREN','CNT_FAM_MEMBERS','AMT_INCOME_RANGE','AMT_CREDIT_RANGE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
         'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','REGION_POPULATION_RELATIVE','YEARS_BIRTH','YEARS_EMPLOYED','OCCUPATION_TYPE',
         'REGION_RATING_CLIENT','EXT_SOURCE_2','EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','AMT_CREDIT','AMT_INCOME_TOTAL',
         'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_GOODS_PRICE']].copy()


# In[60]:


app_data_new.shape


# ##### Categories attributes in two variables, 'Continuous columns' as num_cols & 'Categorical columns' as cat_cols

# In[61]:


num_cols= ['TARGET','CNT_CHILDREN','CNT_FAM_MEMBERS','REGION_POPULATION_RELATIVE','YEARS_BIRTH','YEARS_EMPLOYED',
         'REGION_RATING_CLIENT','EXT_SOURCE_2','EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE',
          'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','AMT_GOODS_PRICE','AMT_CREDIT','AMT_INCOME_TOTAL']
cat_cols= ['CODE_GENDER','NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE','OCCUPATION_TYPE',
           'AMT_INCOME_RANGE','AMT_CREDIT_RANGE']


# In[62]:


print(len(num_cols)+len(cat_cols))


# ### 3(c). Checking outliers

# #### For finding outliers, we will use box plot.
# #### Boxplot
# * It is used to see quartile-wise distribution for any continuous variable
# * It is also used to see the whether outliers are present in the data or not
# * It is used to see quartile-wise distribution for any continuous variable against a categorical variable
# 
# * Left line of box --> 25th Percentile (Q1)
# * Right line of box --> 75th Percentile (Q3)
# * Middle line of box --> 50th Percentile (Median) (Q2)
# * IQR (Inter Quartile Range) = Q3 - Q1 = 75th Percentile - 25th Percentile
# * Lower Whiskers --> Q1-1.5*(Q3-Q1) --> Q1-1.5* IQR (Emperical Relationships)
# * Upper Whiskers --> Q3+1.5*(Q3-Q1) --> Q3+1.5* IQR (Emperical Relationships)

# #### Outliers are values which lies above the 'Upper whisker' or below the 'Lower whisker'

# In[63]:


for i in num_cols:
    plt.figure(figsize=(8,5))
    sns.boxplot(app_data_new[i])
    plt.xlabel(i)
    plt.title("BOX PLOT OF "+i)
    plt.xticks(rotation=0)
    plt.grid(True)
    plt.show()


# In[64]:


sns.boxplot(app_data_new['CNT_FAM_MEMBERS'])
plt.title('box plot of CNT_FAM_MEMBERS')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[65]:


app_data_new['CNT_FAM_MEMBERS'].describe()


# In[66]:


sns.boxplot(app_data_new['CNT_CHILDREN'])
plt.title('box plot of CNT_CHILDREN')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[67]:


app_data_new['CNT_CHILDREN'].describe()


# In[68]:


app_data_new['CNT_CHILDREN'].value_counts()


# In[69]:


sns.boxplot(app_data_new['YEARS_BIRTH'])
plt.title('box plot of YEARS_BIRTH')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[70]:


sns.boxplot(app_data_new['EXT_SOURCE_2'])
plt.title('box plot of EXT_SOURCE_2')
plt.xticks(rotation=0)
plt.grid(True)
plt.show() 


# In[71]:


sns.boxplot(app_data_new['EXT_SOURCE_3'])
plt.title('box plot of EXT_SOURCE_3')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[72]:


sns.boxplot(app_data_new['OBS_30_CNT_SOCIAL_CIRCLE'])
plt.title('box plot of OBS_30_CNT_SOCIAL_CIRCLE')
plt.xticks(rotation=0)
plt.grid(True)
plt.show() 


# In[73]:


sns.boxplot(app_data_new['DEF_30_CNT_SOCIAL_CIRCLE'])
plt.title('box plot of DEF_30_CNT_SOCIAL_CIRCLE')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[74]:


sns.boxplot(app_data_new['OBS_60_CNT_SOCIAL_CIRCLE'])
plt.title('box plot of OBS_60_CNT_SOCIAL_CIRCLE')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[75]:


sns.boxplot(app_data_new['DEF_60_CNT_SOCIAL_CIRCLE'])
plt.title('box plot of DEF_60_CNT_SOCIAL_CIRCLE')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[76]:


sns.boxplot(app_data_new['AMT_GOODS_PRICE'])
plt.title('box plot of AMT_GOODS_PRICE')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[77]:


app_data_new['AMT_GOODS_PRICE'].describe()


# ### 4. Imbalance checking for target columns 'TARGET'.
# ##### Target variable 
# 
# #### 1 - (Defaulter) client with payment difficulties
# (he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample)
# 
# #### 0 - (Not- Defaulter) all other cases
# (All other cases when the payment is paid on time)

# In[78]:


app_data_new['TARGET'].value_counts(normalize= True)*100


# In[79]:


# Create new Data-frame for 'Target'-0 & 1
app_data_target0= app_data_new[app_data_new['TARGET']==0]
app_data_target1= app_data_new[app_data_new['TARGET']==1]


# In[80]:


app_data_new['TARGET'].value_counts(normalize= True).plot.bar(color=('orange','b'))
plt.grid(True)
plt.xticks(rotation=0)
plt.xlabel('Target variables')
plt.ylabel('Count in percentage')
plt.title('Percentage of Defaulter vs Non-defaulter')


# In[81]:


app_data_target0.shape


# In[82]:


app_data_target1.shape


# In[83]:


#### Data imbalance ratio


# In[84]:


Ratio_of_0_1= app_data_target0.shape[0]/app_data_target1.shape[0]
print(Ratio_of_0_1)


#  #### In application_data there exists 92.22% of "not default" and 7.78% of "default" customers. Ratio for non-defaulter(0) & defaulter(1) is 11.856, which represents lots of data imbalance in 'TARGET' data.

# ### 5. Analysis

# ####  5(a). Uni-variate Analysis

# In[85]:


plt.figure(figsize=(22,8))

plt.subplot(1,2,1)
sns.countplot(app_data_target0['OCCUPATION_TYPE'])
plt.xticks(rotation=90)
plt.title('Non-Defaulter Customers')

plt.subplot(1,2,2)
sns.countplot(app_data_target1['OCCUPATION_TYPE'])
plt.xticks(rotation=90)
plt.title('Defaulter Customers')
plt.show()


# * From the count plot we can see that, for a large portion of the people who applied for the loan, there occupation details
#   were missing in records. 
# * In both 'Defaulter' & 'Non-defaulter' category, Laborers are the highest followed by Sales staffs.

# #### Housing related info of Loan Applicants

# In[86]:


plt.figure(figsize=(15,10))

plt.subplot(1,2,1)
sns.countplot(app_data_target1['NAME_HOUSING_TYPE'])
plt.xticks(rotation=20)
plt.title('DEFAULTERS_HOUSING_TYPE')

plt.subplot(1,2,2)
sns.countplot(app_data_target0['NAME_HOUSING_TYPE'])
plt.xticks(rotation=20)
plt.title('NON-DEFAULTERS_HOUSING_TYPE')
plt.show()


# * For both Defaulters & Non-defaulters, House/apartment is having the highest counts amongst all 'Housing_Types'. 

# In[87]:


def value_wise_defaulters_percentage(df, col):
    new_df= pd.DataFrame(columns=['value','Percentage of Defaulter'])
    
    for value in df[col].unique():
        default_cnt=df[(df[col]==value) & (df.TARGET==1)].shape[0]
        total_cnt= df[df[col]==value].shape[0]
        new_df= new_df.append({'value': value,'Percentage of Defaulter': (default_cnt*100/total_cnt)}, ignore_index=True)
    return new_df.sort_values(by='Percentage of Defaulter', ascending= False)


# In[88]:


value_wise_defaulters_percentage(app_data,'NAME_HOUSING_TYPE')


# * Most of the defaulters lives in Rented apartment(12.05%) & with parents (11.47%) and least defaulters live in 'Office apartments'.
# * This data represents, working or reacher applicants are less likely to be defaulters. 

# #### Occupation & Education related info for loan applicants

# In[89]:


plt.figure(figsize=(20,30))

plt.subplot(1,2,1)
app_data_target0['NAME_EDUCATION_TYPE'].value_counts(normalize=True).plot.pie()
plt.title('NON-DEFAULTER')

plt.subplot(1,2,2)
app_data_target1['NAME_EDUCATION_TYPE'].value_counts(normalize=True).plot.pie()
plt.title('DEFAULTER')
plt.show()


# *Above pie chart shows that, for both 'Defaulter' & 'Non-defaulter', Secondary/Secondary special, Education types are highest followed by 'Higher education' and 'Lower secondary' are least in both. 

# In[90]:


value_wise_defaulters_percentage(app_data,'NAME_EDUCATION_TYPE')


# * Applicants with education upto 'Lower secondary' have highest defaulting percentage, however for 'Academic degree', default percentage is least. 

# In[91]:


fig= plt.figure(figsize= (12,5))
sns.countplot(app_data['NAME_INCOME_TYPE'], hue=app_data['TARGET'])
plt.title('Income type counts for defaulters & non-defaulters')
plt.xticks(rotation=0)
plt.show()


# In[92]:


application_data['NAME_INCOME_TYPE'].value_counts(normalize=True)*100


# In[93]:


value_wise_defaulters_percentage(application_data,'NAME_INCOME_TYPE')


# * Majority of loan applicants are working
# * Highest defaulters are those applicants who are either on 'Maternity leave' or 'Unemployed'.
# * 'Students' & 'Businessman' have least default percentage, i.e. 0%. 
# * Employees of income types, Maternity leave, Businessman, Student, Unemployed are very few in counts to contribute in the analysis. 

# In[94]:


value_wise_defaulters_percentage(app_data,'OCCUPATION_TYPE')


# *  Low-skill Laborers have highest defaulting percentage followed by Drivers. Least defaulting percentage are for 'Accountants'. 

# #### Social Circle related info for loan applicants

# In[95]:


start_idx= app_data.columns.get_loc('OBS_30_CNT_SOCIAL_CIRCLE')
end_idx= app_data.columns.get_loc('DEF_60_CNT_SOCIAL_CIRCLE')

social_circle= app_data.iloc[:, start_idx:end_idx+1]


# In[96]:


social_circle.describe()


# In[97]:


social_circle.info()


# In[98]:


plt.figure(figsize=(10,6))
sns.heatmap(social_circle.corr(), annot=True, cmap='Greens')
plt.title('Co-relation matrix for Social_circle')


# * OBS_30_CNT_SOCIAL_CIRCLE and OBS_60_CNT_SOCIAL_CIRCLE are ideantical. 
# * There is high co-relation between DEF_30_CNT_SOCIAL_CIRCLE and DEF_60_CNT_SOCIAL_CIRCLE. 

# In[99]:


fig= plt.subplots(figsize= (15,10))

for i,j in enumerate(['DEF_60_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']):
    plt.subplot(2,2, i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(j, data=app_data_target1)
    plt.title('Defaulters')
    plt.xticks(rotation=90)
    plt.tight_layout()


# In[100]:


fig= plt.subplots(figsize= (15,10))

for i,j in enumerate(['DEF_60_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']):
    plt.subplot(2,2, i+1)
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(j, data=app_data_target0)
    plt.title('Non-Defaulters')
    plt.xticks(rotation=90)
    plt.tight_layout()


# * For both Defaulters & Non-Defaulters, 'DEF_60_CNT_SOCIAL_CIRCLE' & 'OBS_60_CNT_SOCIAL_CIRCLE features are showing similar trends.

# #### Asset Related details

# In[101]:


app_data[['FLAG_OWN_CAR','FLAG_OWN_REALTY','TARGET']].info()


# In[102]:


fig= plt.figure(figsize= (14,5))

ax1= fig.add_subplot(1,2,1, ylim=(0,200000))
sns.countplot(app_data['FLAG_OWN_REALTY'], hue=app_data['TARGET'], order=['Y','N'], ax=ax1)
plt.title('FLAG_OWN_REALTY for Defaulter/Non-defaulter')

ax2= fig.add_subplot(1,2,2, ylim=(0,200000))
sns.countplot(app_data['FLAG_OWN_CAR'], hue=app_data['TARGET'], order=['Y','N'], ax=ax2)
plt.title('FLAG_OWN_CAR for Defaulter/Non-defaulter')

plt.show()


# * Most of the loan applicants not own CAR. 
# * Most of the loan applicants own REALTY. 

# In[103]:


value_wise_defaulters_percentage(app_data, 'FLAG_OWN_REALTY')


# In[104]:


value_wise_defaulters_percentage(app_data, 'FLAG_OWN_CAR')


# * Numbers of defaulters are larger for those who does not own REALTY & CAR than who owns REALTY & CAR.

# #### Family related info for loan applicants

# In[105]:


app_data[['CNT_FAM_MEMBERS','CNT_CHILDREN','NAME_FAMILY_STATUS']].info()


# In[106]:


fig= plt.subplots(figsize= (15,15))

for i,j in enumerate(['CNT_FAM_MEMBERS','CNT_CHILDREN','NAME_FAMILY_STATUS']):
    plt.subplot(3,3,i+1, ylim=(0,200000))
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(app_data[j], hue=app_data['TARGET'])
    plt.xticks(rotation=90)
    plt.tight_layout()


# In[107]:


value_wise_defaulters_percentage(app_data, 'CNT_CHILDREN')


# * Applicants having larger numbers of Children counts in the family are maximum in defaulters category. 

# In[108]:


value_wise_defaulters_percentage(app_data, 'NAME_FAMILY_STATUS')


# * For Civil marriage &  single/not married applicants, default rate is highest and least for Widows. 

# In[109]:


value_wise_defaulters_percentage(app_data, 'CNT_FAM_MEMBERS')


# * Applicants having larger number of family members are maximum in defaulters category. 

# #### Gender related information for loan applicants

# In[110]:


sns.countplot(app_data['CODE_GENDER'], hue=app_data['TARGET'])
plt.show()


# In[111]:


value_wise_defaulters_percentage(app_data, 'CODE_GENDER')


# * Defaulter percentage is higher for Male applicants than Female applicants. 
# * Female applicants count is higher than male applicants. 

# In[112]:


fig= plt.figure(figsize=(15,5))

ax1= fig.add_subplot(1,2,1, title= 'Defaulter')
sns.kdeplot(app_data[app_data['TARGET']==1]['YEARS_BIRTH'], ax=ax1)

ax2= fig.add_subplot(1,2,2, title= 'Non_defaulter')
sns.kdeplot(app_data[app_data['TARGET']==0]['YEARS_BIRTH'], ax=ax2)

plt.grid(True)
plt.show()


# * For elder people (age>40 years), default percentages are less.
# * Highest defaulting percentage is for 30 years age applicants. 

# #### Income & annuity related info for loan applicants

# In[113]:


plt.figure(figsize=(10,2))
sns.boxplot(app_data['AMT_INCOME_TOTAL'])
plt.show()


# In[114]:


plt.figure(figsize=(10,2))
sns.boxplot(app_data['AMT_ANNUITY'])
plt.show()


# * Boxplot representing that there are lots of outliers for Income & Annuity. These outliers will bias the further analysis. So, excluding the values outside the 99 percentile for AMT_ANNUITY & AMT_INCOME_TOTAL.  

# In[115]:


app_data=app_data[app_data['AMT_ANNUITY']< np.nanpercentile(app_data['AMT_ANNUITY'], 99)]
app_data=app_data[app_data['AMT_INCOME_TOTAL']< np.nanpercentile(app_data['AMT_INCOME_TOTAL'], 99)]


# In[116]:


fig= plt.figure(figsize=(15,5))

ax1= fig.add_subplot(1,2,1, title= 'Defaulter')
sns.kdeplot(app_data[app_data['TARGET']==1]['AMT_ANNUITY'], ax=ax1)

ax2= fig.add_subplot(1,2,2, title= 'Non_defaulter')
sns.kdeplot(app_data[app_data['TARGET']==0]['AMT_ANNUITY'], ax=ax2)

plt.grid(True)
plt.show()


# * AMT_ANNUITY distribution is similar for both Defaulters & Non-defaulters. 

# In[117]:


fig= plt.figure(figsize=(20,6))

ax1= fig.add_subplot(1,2,1, title= 'Defaulter')
sns.kdeplot(app_data[app_data['TARGET']==1]['AMT_INCOME_TOTAL'], ax=ax1)

ax2= fig.add_subplot(1,2,2, title= 'Non_defaulter')
sns.kdeplot(app_data[app_data['TARGET']==0]['AMT_INCOME_TOTAL'], ax=ax2)

plt.show()


# #### AMT_GOODS_PRICE related info for loan applicants

# In[118]:


plt.figure(figsize=(22,10))

plt.subplot(1,2,1)
plt.style.use("tableau-colorblind10")
sns.distplot(app_data_target0['AMT_GOODS_PRICE'], bins=40, color='r')
plt.xticks(rotation=20)
plt.title('NON_DEFAULTERS_AMT_GOODS_PRICE')

plt.subplot(1,2,2)
plt.style.use("tableau-colorblind10")
sns.distplot(app_data_target1['AMT_GOODS_PRICE'], bins=40, color='r')
plt.xticks(rotation=20)
plt.title('DEFAULTERS_AMT_GOODS_PRICE')


# * The distplot representing the univariate density distribution of data i.e. 'AMT_GOODS_PRICE' for both defaulters & non-defaulters.

# #### 5(b). Bi-variate & Multi-Variate analysis

# In[119]:


fig= plt.figure(figsize=(20,6))

ax1= fig.add_subplot(1,2,1, title= 'Defaulter')
sns.scatterplot(app_data[app_data['TARGET']==1]['AMT_GOODS_PRICE'],app_data[app_data['TARGET']==1]['AMT_CREDIT'], ax=ax1)

ax2= fig.add_subplot(1,2,2, title= 'Non_defaulter')
sns.scatterplot(app_data[app_data['TARGET']==1]['AMT_GOODS_PRICE'],app_data[app_data['TARGET']==1]['AMT_CREDIT'], ax=ax2)

plt.ticklabel_format(style='plain', axis='x')
plt.ticklabel_format(style='plain', axis='y')

plt.show()


# * 'AMT_GOODS_PRICE' & 'AMT_CREDIT' has linear relationship.

# #### 1. Finding top 10 co-relation for Defaulters.

# In[120]:


default_corr= app_data_target1.corr()
round(default_corr,3)

corr_list1= default_corr.unstack()
corr_list1.sort_values(ascending= False).drop_duplicates().head(11)


# In[121]:


plt.figure(figsize=(10,6)) 
sns.heatmap(default_corr, annot=True, cmap='coolwarm')
plt.title('Correlation matrix for target variable 1')
plt.show()


# #### 2. Finding top 10 co-relation for Non-Defaulters.

# In[122]:


non_default_corr= app_data_target0.corr()
round(non_default_corr,3)

corr_list2= non_default_corr.unstack()
corr_list2.sort_values(ascending= False).drop_duplicates().head(11)


# In[123]:


plt.figure(figsize=(10,6)) 
sns.heatmap(non_default_corr, annot=True, cmap="coolwarm")
plt.title('Correlation matrix for target variable 0')
plt.show()


# In[124]:


g= sns.pairplot(app_data[num_cols], height=2.5, diag_kind = 'kde')
g.fig.suptitle("Pair Plot for Continuous variables")


# ### Analyzing 'previous_application' datasets

# #### 1. Loading datasets

# In[125]:


previous_app= pd.read_csv('previous_application.csv')


# In[126]:


previous_app.head()


# #### 2. Data Pre-processing

# In[127]:


previous_app.shape


# In[128]:


previous_app.describe()


# In[129]:


previous_app.info()


# #### 3. Handling missing values

# In[130]:


previous_app.isnull().sum()/len(previous_app.index)*100


# #### Deleted the features with missing values>=90%, as such large no of missing values will impact the analysis.

# In[131]:


prev_app=previous_app.drop(['RATE_INTEREST_PRIMARY','RATE_INTEREST_PRIVILEGED'], axis=1)


# In[132]:


prev_app.isnull().sum()/len(prev_app.index)*100


# #### 'PRODUCT_COMBINATION' & 'AMT_CREDIT' has very low percentage (~2%) of missing data. Dropping entries (rows) would not impact our analysis and wont loose the data. 

# In[133]:


prev_app.dropna(subset= ['PRODUCT_COMBINATION','AMT_CREDIT'], inplace=True)


# In[134]:


prev_app['AMT_GOODS_PRICE'].value_counts()


# In[135]:


sns.boxplot(prev_app['AMT_GOODS_PRICE'])
plt.show()


# #### From the box plot, we can see that there are large no of outliers present in the 'AMT_GOODS_PRICE'

# In[136]:


prev_app['AMT_GOODS_PRICE'].median()


# In[137]:


prev_app['CNT_PAYMENT'].value_counts()


# In[138]:


sns.boxplot(prev_app['CNT_PAYMENT'])
plt.show()


# #### From the box plot, we can see that there are outliers present in the 'CNT_PAYMENT'

# In[139]:


prev_app['AMT_ANNUITY'].value_counts()


# In[140]:


sns.boxplot(prev_app['AMT_ANNUITY'])
plt.show()


# #### From the box plot, we can see that there are large no of outliers present in the 'AMT_ANNUITY'

# In[141]:


prev_app.info()


# In[142]:


prev_app['NAME_TYPE_SUITE'].value_counts()


# #### Separating numerical features from previous_appliation data.

# In[143]:


num_col1= []
for col in prev_app.columns:
    if prev_app[col].dtype==int or prev_app[col].dtype==float:
        num_col1.append(col)
        
num_col1


# In[144]:


prev_app_num=pd.DataFrame()

for col in num_col1:
    prev_app_num[col]=prev_app[col]
prev_app_num.info()


# #### Co-relation checking b/w numerical features for Previous application data

# In[145]:


plt.figure(figsize=(10,6))
sns.heatmap(prev_app_num.corr(), annot=True)
plt.title('Co-relation checking b/w numerical features for Previous application data')
plt.show()


# * We can see high corelation between 'AMT_ANNUITY' & 'AMT_APPLICATION', 'AMT_CREDIT' & 'AMT_GOODS_PRICE'. 

# #### 4. Merging both files 'APPLICATION_DATA' & 'PREVIOUS_APPLICATION'

# In[146]:


merge_data= pd.merge(left= application_data, right=previous_app, how='inner', on='SK_ID_CURR', suffixes='_x')


# In[147]:


merge_data.head()


# In[148]:


merge_data.shape


# In[149]:


merge_data.describe()


# In[150]:


merge_data.dtypes


# #### 5. Uni-variate, Bi-variate & Multi-variate analysis

# In[151]:


merge_data['NAME_CONTRACT_STATUS'].value_counts()/len(merge_data.index)*100


# In[152]:


plt.figure(figsize=(10,5))
sns.countplot(merge_data['NAME_CONTRACT_STATUS'])
plt.xlabel('NAME_CONTRACT_STATUS')
plt.ylabel('COUNTS')
plt.title('Count plot for NAME_CONTRACT_STATUS')
plt.show()


# #### Percentage of Previously approved loan applicant who defaulted in current loan. 

# In[153]:


Approved_total= merge_data[merge_data['NAME_CONTRACT_STATUS']=='Approved'].shape[0]
Approved_defaulter= merge_data[(merge_data['TARGET']==1) & (merge_data['NAME_CONTRACT_STATUS']=='Approved')].shape[0]

print('Percentage of previously approved loan applicants who defaulted in current loan:', (Approved_defaulter*100/Approved_total))


# #### Percentage of previously refused loan applicants who were Non-defaulter in current loan

# In[154]:


refused_total= merge_data[merge_data['NAME_CONTRACT_STATUS']=='Refused'].shape[0]
refused_non_default= merge_data[(merge_data['TARGET']==0) & (merge_data['NAME_CONTRACT_STATUS']=='Refused')].shape[0]

print('Percentage of previously refused loan applicants who were Non-defaulters in current loan:', (refused_non_default*100/refused_total))


# In[155]:


plt.style.use('ggplot')
plt.title('Status of previous loan application')
sns.countplot(merge_data['NAME_CONTRACT_STATUS'], hue= merge_data['TARGET'])
plt.show()


# #### Observation:
# * Loan applicants whose previous loan application was rejected are more likely to pay current loan on time. 
# * 7.6% applicant whose previous loan was 'Approved' were defaulter in current loan.
# * 88% applicants whose previous loan was 'Rejected' were non-defaulters (able to pay amount on time) in current loan

# In[156]:


fig= plt.figure(figsize=(12,6))

ax1= fig.add_subplot(1,2,1, ylim=(0,70000), title= 'Non_defaulter')
sns.scatterplot(merge_data[merge_data['TARGET']==0]['AMT_ANNUITY_'], merge_data[merge_data['TARGET']==0]['AMT_DOWN_PAYMENT'])

ax2= fig.add_subplot(1,2,2, ylim=(0,70000), title= 'Defaulter')
sns.scatterplot(merge_data[merge_data['TARGET']==1]['AMT_ANNUITY_'], merge_data[merge_data['TARGET']==1]['AMT_DOWN_PAYMENT'])

plt.show()


# #### Observation:
# * Defaulter cases are less for Higher down payment for loan applicants. 
# * Defaulters numbers are less compared to Non-defaulters for larger AMT_ANNUITY of previous_application

# #### Analysis of Categorical Features of Previous Application data

# In[157]:


plt.figure(figsize=(8,5))
sns.countplot(merge_data['NAME_CONTRACT_TYPE_'], hue=merge_data.TARGET)
plt.title('NAME_CONTRACT_TYPE for Target variables(0&1)')
plt.show()


# #### Observation:
# * Highest nos of loan applied are 'Cash loans' for all applicants. So, maximum no of both categories, i.e. Defaulter & Non-defaulter are for cash loans. 

# In[158]:


fig= plt.figure(figsize=(12,6))

ax1= fig.add_subplot(1,2,1, title= 'Non_defaulter')
merge_data[merge_data['TARGET']==0]['HOUR_APPR_PROCESS_START_'].hist(bins=10, ax=ax1)

ax2= fig.add_subplot(1,2,2, title= 'defaulter')
merge_data[merge_data['TARGET']==1]['HOUR_APPR_PROCESS_START_'].hist(bins=10, ax=ax2)

plt.show()


# * Above histplot shows that, most of the loans are applied around 10:00 HRS then 15:00 HRS.This features doesn't impact on 'TARGET' variable.

# In[159]:


plt.figure(figsize=(8,5))
sns.countplot(merge_data[merge_data['NAME_CONTRACT_STATUS']=='Refused']['CODE_REJECT_REASON'], hue= merge_data.TARGET)
plt.title('CODE_REJECT_REASON for Target variables(0 & 1)')
plt.show()


# * In above plot, we can see that 'HC' is the most frequent reason followed by 'LIMIT' & 'SCO' for Loans Rejection of applicants. 

# ### Checking percentage of defaulters of Previous applicants for different categories.

# In[160]:


def value_wise_percentage_of_defaulters(df, col):
    new_df= pd.DataFrame(columns=['value', 'Percentage of Defaulter'])
    
    for value in df[col].unique():
        default_cnt= df[(df[col]==value)& (df.TARGET==1)].shape[0]
        total_cnt= df[df[col]==value].shape[0]
        new_df= new_df.append({'value': value, 'Percentage of Defaulter': (default_cnt*100/total_cnt)}, ignore_index=True)
    return new_df.sort_values(by='Percentage of Defaulter', ascending= False)


# In[161]:


value_wise_percentage_of_defaulters(merge_data, 'NAME_PRODUCT_TYPE')


# * Amongst all NAME_PRODUCT_TYPE category, walk-in defaulted maximum, i.e. 12.5%

# In[162]:


value_wise_percentage_of_defaulters(merge_data, 'NAME_SELLER_INDUSTRY')


# * Highest defaulters are for 'Auto technology' (10.37%) for previous applicants amongst all 'NAME_SELLER_INDUSTRY' & least for 'Tourism'. 

# In[163]:


value_wise_percentage_of_defaulters(merge_data, 'NAME_PORTFOLIO')


# * For previous applicants, defaulter rate is highest for 'Cards' & least for 'Cars'.

# In[164]:


value_wise_percentage_of_defaulters(merge_data, 'NAME_YIELD_GROUP')


# * Highest deafulters percentage is for unknown NAME_YIELD_GROUP, i.e. 9.9% and least for 'low_action',i.e. 6.45%.

# In[165]:


value_wise_percentage_of_defaulters(merge_data, 'NAME_GOODS_CATEGORY')


# * From previous applicants, highest defaulters are those who has previously applied for Insurance and Vehicles and least are those who had applied for Animals followed by Fitness.  

# In[166]:


value_wise_percentage_of_defaulters(merge_data, 'CHANNEL_TYPE')


# * Almost 12.9% previous applicants defaulted for AP+ (Cash loan), which is highest amongst all 'CHANNEL_TYPE'.

# ### Conclusions:
# * In application_data there exists 92.22% of "not default" and 7.78% of "default" customers. The ratio for non-defaulter (0) & defaulter (1) is 11.856, which represents lots of data imbalance in 'TARGET' data.
# * In the both 'Defaulter' & 'Non-defaulter' categories, Labourers are the highest followed by Sales staff.
# * Most of the defaulters live in a rented apartment (12.05%) & with parents (11.47%) and the least defaulters live in 'Office apartments'.
# * Working or richer applicants are less likely to be defaulters.
# * Majority of loan applicants are working
# * Highest defaulters are those applicants who are either on 'Maternity leave' or 'Unemployed'.
# * 'Students' & 'Businessman' have the least default percentage, i.e. 0%. .
# * Low-skill Labourers have the highest defaulting percentage followed by Drivers. The least defaulting percentage is for 'Accountants'.
# * Most of the loan applicants do not own a CAR & most of them own REALTY.
# * Numbers of defaulters are larger for those who do not own REALTY & CAR than those who own REALTY & CAR.
# * Applicants having larger numbers of Children counts in the family are maximum in the defaulter’s category.
# * For Civil marriage &  single /not married applicants, the default rate is highest and least for Widows.
# * Applicants having a larger number of family members are maximum in the defaulter’s category.
# * Defaulter percentage is higher for Male applicants than Female applicants. 
# * Female applicant count is higher than male applicants.
# * For elder people (age>40 years), default percentages are less.
# * Highest defaulting percentage is for 30 years age applicants.
# * Percentage of previously refused loan applicants who were Non-defaulters in current loan: 88.00358612820408
# * Percentage of previously approved loan applicants who defaulted on current loan: 7.588655443691958
# * Loan applicants whose previous loan application was rejected are more likely to pay the current loan on time. 
# * 7.6% of an applicant whose previous loan was 'Approved' were defaulters in the current loan.
# * 88% of applicants whose previous loan was 'Rejected' were non-defaulters (able to pay the amount on time) in the current loan
# * Highest no of loans applied is 'Cash loans' for all applicants. So, the maximum no of both categories, i.e. Defaulter & Non-defaulter are for cash loans.
# * 'HC' is the most frequent reason followed by 'LIMIT' & 'SCO' for Loans Rejection of applicants.
# * Amongst all NAME_PRODUCT_TYPE categories, walk-in defaulted maximum, i.e. 12.5%
# * Highest defaulters are for 'Auto technology' (10.37%) for previous applicants amongst all 'NAME_SELLER_INDUSTRY' & least for 'Tourism'.
# * For previous applicants, the defaulter rate is highest for 'Cards' & least for 'Cars'.
# * Highest defaulter percentage is for unknown NAME_YIELD_GROUP, i.e. 9.9% and least for 'low_action',i.e. 6.45%.
# * From previous applicants, the highest defaulters are those who have previously applied for Insurance and Vehicles and the least are those who had applied for Animals followed by Fitness.  
# * Almost 12.9% of previous applicants defaulted for AP+ (Cash loan), which is the highest amongst all 'CHANNEL_TYPE'.

# ### Thank you!
