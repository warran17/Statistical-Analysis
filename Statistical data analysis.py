#!/usr/bin/env python
# coding: utf-8

# ##  Project: Statistical Data Analysis <a class="tocSkip">
# 
# 
# **Project Description:**
# <div >
#     
# > Clients behaviour analysis of the telecom operator megaline's client.    
# 
# </div>
# 
# **Goal of the Project:**
#     
# <div >
#     
# > To determine which plan made more revenue per month and compare revenue from users in NY-NJ area to users in other area.    
# 
# </div>
# 
# 

# ##  Table of Contents <a class="tocSkip">
# 
# **Step 1. Open the data file and study the general information**
# 
# **Step 2. Prepare the data**
# 
# **Step 3. Analyze the data**
# 
# **Step 4. Test the hypotheses**
# 
#     4.1  Hypotheses test (based on plan)
#     
#     4.2  Area wise hypothesis test
# 
# **Step 5. Overall Conclusion**

# <h2> Step 1. Open the data file and study the general information </h2>

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt 
import math
import numpy as np
from scipy import stats as st


# In[2]:


from nltk.stem import SnowballStemmer 

english_stemmer = SnowballStemmer('english')
import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemma = WordNetLemmatizer() 


# In[3]:


df_calls= pd.read_csv('/datasets/megaline_calls.csv')
df_internet= pd.read_csv('/datasets/megaline_internet.csv')
df_messages= pd.read_csv('/datasets/megaline_messages.csv')
df_plans= pd.read_csv('/datasets/megaline_plans.csv')
df_users= pd.read_csv('/datasets/megaline_users.csv')


# In[4]:


df_calls.head()


# In[5]:


df_calls.info()


# In[6]:


print(df_calls[df_calls.duplicated()])


# In[7]:


print(df_messages[df_messages.duplicated()])


# In[8]:


print(df_internet[df_internet.duplicated()])


# In[9]:


df_messages.info()


# In[10]:


df_internet.info()


# In[11]:


df_internet.head()


# In[12]:


df_messages.head()


# In[13]:


df_plans.head()


# In[14]:


df_users.head(30)


# ## Conclusion <a class="tocSkip">
# 
# >No missing and duplicated data
# 

# <h2> Step 2. Prepare the data </h2>

# In[15]:


df_calls['duration']=np.ceil(df_calls['duration']).astype(int)
df_calls.head()


# > The call duration should have rounded up in each call

# In[16]:


df_calls['call_date'] = pd.to_datetime(df_calls['call_date'], format='%Y-%m-%d')
df_messages['message_date'] = pd.to_datetime(df_messages['message_date'], format='%Y-%m-%d')
df_internet['session_date'] = pd.to_datetime(df_internet['session_date'], format='%Y-%m-%d')
df_calls['month']=df_calls['call_date'].dt.month
df_messages['month']=df_messages['message_date'].dt.month
df_internet['month']=df_internet['session_date'].dt.month


# >All the dates should be in datetime format in order to process them. Also, since revenue are caollected in monthly basis it
# is better to make separate month column

# In[17]:


pivot_calls= pd.pivot_table(df_calls , index=['user_id','month'], values=['duration'],
                    aggfunc={'count','mean'})
print(pivot_calls)


# > Average call duration and number of calls made by each user in each month are obtained

# In[18]:


pivot_messages= pd.pivot_table(df_messages , index=['user_id','month'], values=['id'],
                    aggfunc={'count'})
print(pivot_messages)


# > Number of messages sent by each user in each month are obtained

# In[19]:


pivot_internet= pd.pivot_table(df_internet , index=['user_id','month'], values=['mb_used'], aggfunc='sum'
                    )
print(pivot_internet)


# In[20]:


import sys
import warnings
if not sys.warnoptions:
       warnings.simplefilter("ignore")


# In[21]:


total1= pd.merge(left = pivot_calls , right = pivot_messages, how='outer',on=['user_id', 'month']).fillna(0)
total2=pd.merge(left = total1 , right = pivot_internet, how='outer',on=['user_id', 'month']).fillna(0)
print(total2.head(20))


# In[22]:


#new_data= df_users [['user_id','plan']]


# In[23]:


total_final=pd.merge(left = df_users , right =total2, how='outer',on=['user_id']).fillna(0)


# In[24]:


print(total_final)


# In[25]:


total_final.columns=['user_id', 'first_name',   'last_name',  'age', 'city' ,   'reg_date', 'plan', 'churn_date', 'no_calls','avg_duration','messages', 'data_mb']


# In[26]:


print(total_final.head())


# In[27]:


total_final['data_mb']= np.ceil(total_final['data_mb']).astype(int)
total_final['monthly_call']=total_final['no_calls']*total_final['avg_duration'].astype(int)


# In[28]:


print(total_final.head(2))


# *Average call duration and number of calls, number of messages and data used by each user in each month are obtained.
# These data are merged with user's plan.
# 
# *Column names are changed in appropiate way than in merged table which had columns name as in pivot tables.
# Likewise, the monthly data is rounded up charged applied in such a way.

# In[29]:


total_final.loc[total_final.plan=='surf', 'revenue']=20
total_final.loc[total_final.plan=='ultimate', 'revenue']=70

#total_final.loc[total_final.plan=='ultimate' & total_final.plan>1000, 'revenue']+= (total_final.plan-1000)*0.01

def revenue(row):
    if row['plan']=='surf':
        if row['monthly_call']> 500:
            call_cost= (row['monthly_call']- 500)*0.03
        if row['monthly_call']<= 500:
            call_cost=0
        if row['messages']> 50:
            msg_cost= (row['messages']- 50)*0.03
        if row['messages']<= 50:
            msg_cost=0
        if row['data_mb']> 15*1024:
            data_cost= (row['data_mb']- 15*1024)*10/1024
        if row['data_mb']<= 15*1024:
            data_cost=0
        return row['revenue']+call_cost+ msg_cost+ data_cost
    elif row['plan']=='ultimate':
        if row['monthly_call']> 3000:
            call_cost= (row['monthly_call']- 3000)*0.01
        if row['monthly_call']<= 3000:
            call_cost=0
        if row['messages']> 1000:
            msg_cost= (row['messages']- 1000)*0.01
        if row['messages']<= 1000:
            msg_cost=0
        if row['data_mb']> 30*1024:
            data_cost= (row['data_mb']- 30*1024)*7/1024
        if row['data_mb']<= 30*1024:
            data_cost=0
        return row['revenue']+call_cost+ msg_cost+ data_cost
    
total_final['revenue'] = total_final.apply(revenue, axis=1)
print(total_final.head(10)) 


# In[30]:


zz = np.abs(st.zscore(total_final['revenue']));
revenue = total_final[(zz < 3)];
plt.hist(revenue['revenue'], density= True, alpha=0.7, bins=100);

plt.title('Density plot of revenue');
plt.xlabel('monthly revenue ($) per user');
plt.ylabel('density');


# Two spikes can be seen at about base prices of both plan suggest, people try to use service within the free service provided in corresponsing plan.

# ## Conclusion <a class="tocSkip">

# >Revenue per month is calculated for each user.
# 
# >Average call duration and number of calls, number of messages and data used by each user in each month are obtained. These data are merged with user's plan.

# <h2> Step 3. Analyze the data </h2>

# In[31]:


plt.hist(total_final.loc[total_final['plan']=='surf','revenue'], density= True, alpha=0.7)
plt.hist(total_final.loc[total_final['plan']=='ultimate','revenue'], density= True, alpha=0.3)
plt.title('Monthly call duration in both  plan')
plt.xlabel('monthly call duration (minutes per user)')
plt.ylabel('density')


# In[32]:


print('Maximum revenue ( surf plan) : {:.2f}'.format(total_final.loc[total_final['plan']=='surf','revenue'].max()))
print('Maximum revenue ( ultimate plan) : {:.2f}'.format(total_final.loc[total_final['plan']=='ultimate','revenue'].max()))


# > About 90% of the user in ultimate plan spent 70 euro per month and maximum of it goes upto 178.5 euro per month
# 
# >In surf plan, about 17% users spent 20-70 euro per month. Likewise there are some users who spent upto 250 euro per month.  Also, the maximum amount spent by surf plan user is 580 euro.

# In[33]:


surf=total_final[total_final['plan']=='surf']
ultimate=total_final[total_final['plan']=='ultimate']
z1 = np.abs(st.zscore(ultimate[['messages','data_mb','monthly_call','revenue']]))
z2 = np.abs(st.zscore(surf[['messages','data_mb','monthly_call','revenue']]))
ultimate = ultimate[(z1 < 3).all(axis=1)]
surf = surf[(z2 < 3).all(axis=1)]


# In[34]:


print(total_final.groupby('plan')[['messages','data_mb','monthly_call']].sum().astype(int))


# >Minutes, texts, and volume of data the users of each plan require per month are otained
# 
# >Minutes, texts, and volume of data require in ultimate plan are about half of that required for surf plan users.

# In[35]:


#print(total_final.groupby('plan')[['messages','data_mb','monthly_call']].mean())
print('Surf')
print(surf[['messages','data_mb','monthly_call','revenue']].mean().astype(int))
print('Ultimate')
print(ultimate[['messages','data_mb','monthly_call','revenue']].mean().astype(int))


# In[36]:


#print(total_final.groupby('plan')[['messages','data_mb','monthly_call']].std())
print('Surf')
print(surf[['messages','data_mb','monthly_call','revenue']].std().astype(int))
print('Ultimate')
print(ultimate[['messages','data_mb','monthly_call','revenue']].std().astype(int))


# In[37]:


#print(total_final.groupby('plan')[['messages','data_mb','monthly_call']].var())
print('Surf')
print(surf[['messages','data_mb','monthly_call','revenue']].var().astype(int))
print('Ultimate')
print(ultimate[['messages','data_mb','monthly_call','revenue']].var().astype(int))


# >Average data used and messages sent seems slightly different in two plans. However, monthly call duration is marginally different.

# >From the t test of average of two population, and average value obtained earlier. It is clear that, 'ultimate' plan user use more dta , 
# sent more messages and use more time on call.

# In[38]:


#plt.hist(total_final.loc[total_final['plan']=='surf','monthly_call'], density= True, alpha=0.7)
#plt.hist(total_final.loc[total_final['plan']=='ultimate','monthly_call'], density= True, alpha=0.3)
plt.hist(surf['monthly_call'], density= True, alpha=0.7);
plt.hist(ultimate['monthly_call'], density= True, alpha=0.3);
plt.title('Monthly call duration in both  plan');
plt.xlabel('monthly call duration (minutes per user)');
plt.ylabel('density');


# >The call duration distribution cannot be distinguised enough with density histogram. However, more proportion of surf plan users 
# call 525 to 630 minutes compared to surf plan users. This is noticeable since, surf plan user get only 500 minutes of free call in packge.

# In[39]:


plt.hist(surf['messages'], density= True, alpha=0.7);
plt.hist(ultimate['messages'], density= True, alpha=0.3);
#plt.hist(total_final.loc[total_final['plan']=='ultimate','messages'], density= True, alpha=0.2, bins=15)
plt.title('Monthly message sent in both  plan');
plt.xlabel('Number of messages per month');
plt.ylabel('density');


# > The significant difference cant be seen in density histogram. However density of message sent after 50 messages 
# decreases quickly for 'surf' plan. This may have relation with number of free messages of the plan. About 80 % 
# ((0.030*13+0.012*13+0.013*13+0.008*13)*100) of user in surf plan sent messages below 50.
# This propertion is slightly smaller in ultimate plan users.

# In[40]:


#plt.hist(total_final.loc[total_final['plan']=='surf','data_mb']/1024, density= True, alpha=0.5, bins=23)
#plt.hist(total_final.loc[total_final['plan']=='ultimate','data_mb']/1024, density= True, alpha=0.2, bins=15)
plt.hist(surf['data_mb']/1024, density= True, alpha=0.7, bins=34)
plt.hist(ultimate['data_mb']/1024, density= True, alpha=0.3, bins=34)
plt.title('Monthly data used in both  plan')
plt.xlabel('data gb')
plt.ylabel('density')


# > Despite having different amount of free data in given packages, maximum propertion of users use 17 gB of data. and the 
# distribution seem to have gaussian in nature. At first look, surf users also use similar amount of data as ultimate plan users.

# In[41]:


plt.hist(surf['revenue'], density= True, alpha=0.7);
plt.hist(ultimate['revenue'], density= True, alpha=0.3);
#plt.hist(total_final.loc[total_final['plan']=='ultimate','messages'], density= True, alpha=0.2, bins=15)
plt.title('Monthly message sent in both  plan');
plt.xlabel('Number of messages per month');
plt.ylabel('density');


# In[42]:


#print(total_final.groupby('plan')[['revenue']].mean())


# In[43]:


print('Surf')
print('Mean:' ,surf['revenue'].mean().astype(int))
print('Variance:' ,surf['revenue'].var().astype(int))
print('Ultimate')
print('Mean:' ,ultimate['revenue'].mean().astype(int))
print('Variance:' ,ultimate['revenue'].var().astype(int))


# ## Conclusion <a class="tocSkip">
# 
# > In histogram, average call duration and number of calls, number of messages and data used by each users have similar 
# distribution in larger extent. Despite having slightly different proportion of users in bins. Maximum proportion of users 
# from each group belong to the same bins in all cases.
# 

# <h2> Step 4. Test the hypotheses </h2>

# <h3>Testing hypothesis (based on plan):</h3>
# 
# >Null Hypothesis: Both plan users generate equal amount of revenue on average
# 
# >alternative Hypothesis: Both plan users generate different amount of revenue on average

# In[44]:


alpha=0.05
results_revenue = st.ttest_ind(
    surf['revenue'],         
    ultimate['revenue'],equal_var = False )

print('p-value:  {:}'.format(results_revenue.pvalue))

if (results_revenue.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis") 


# In[45]:


print(total_final[total_final['city']== 'NY-NJ' ])


# In[46]:


#Split the sentences to lists of words.
total_final['area'] = total_final['city'].str.split()

# Make sure we see the full column.
pd.set_option('display.max_colwidth', -1)
total_final['stemmed']=total_final['area'].apply(lambda x: [english_stemmer.stem(y) for y in x])
   
# Stem every word.
total_final = total_final.drop(columns=['area']) # Get rid of the unstemmed column.


# In[47]:


print(total_final['stemmed'].tail(30))


# In[ ]:





# In[48]:


def user_area(row):
    area = row['stemmed']
    
    
    for query in area:
        for word in query.split(" "):
            #stemmed_word = english_stemmer(word)
            #return stemmed_word
            #if 'new' in stemmed_word:
                #return 'NY-NJ'           
    
            #else :
                #return  'others'  
            if 'york' in word:
                return 'nynj'
            
total_final['users_area'] = total_final.apply(user_area, axis=1) 
total_final['users_area']=total_final['users_area'].fillna('Others')
total_final.tail(30)


# In[49]:


print('mean:',total_final.groupby('users_area')['revenue'].mean());
print('Variance',total_final.groupby('users_area')['revenue'].var());


# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 2:
#     
# Please note that you can avoid "Text(0, 0.5, 'density')" by using ";" after code lines with graph.
# </div>

# In[50]:


plt.hist(total_final.loc[total_final['users_area']=='nynj','revenue'], density= True, alpha=0.2);
plt.hist(total_final.loc[total_final['users_area']=='Others','revenue'], density= True, alpha=0.7);
plt.title('Monthly revenue based on area');
plt.xlabel('revenue');
plt.ylabel('density');


# due to outlier, the most dense area of histogram has only few bins and unable to define characteristic of revenue based on area.

# In[51]:


nynj=total_final[total_final['users_area']=='nynj']
others=total_final[total_final['users_area']=='Others']
z3 = np.abs(st.zscore(nynj['revenue']))
z4 = np.abs(st.zscore(others['revenue']))
nynj = nynj[(z3 < 3)]
others = others[(z4 < 3)]


# In[52]:


plt.hist(nynj['revenue'], density= True, alpha=0.7, bins=11);
plt.hist(others['revenue'], density= True, alpha=0.3, bins=14);
plt.title('Monthly revenue based on area');
plt.xlabel('revenue');
plt.ylabel('density');


# This histogram suggest that there are more proportion of 'ultimate plan' users in other areas than in NY-NJ area and more 'surf plan'
# users in NY-NJ area. so peaks around base prices of each plan are correspondingly lower and higher.

# In[53]:


print('Mean ( ny-nj area) {:.2f}'.format(nynj['revenue'].mean()))
print('Variance( ny-nj area) {:.2f}'.format(nynj['revenue'].var()))
print('mean:( other area) {:.2f}'.format(others['revenue'].mean()))
print('Variance( other area) {:.2f}'.format(others['revenue'].var()))


# <h3>Area wise hypothesis test </h3>
# 
# >Null Hypothesis:  NY-NJ area and other area have equal average revenue collection on averge per user
#     
# >Alternative Hypothesis:  NY-NJ area and other area have different average revenue collection on averge per user

# In[54]:


alpha=0.05
results_revenue = st.ttest_ind(
    nynj['revenue'],         
    others['revenue'] )

print('p-value:  {:.5f}'.format(results_revenue.pvalue))

if (results_revenue.pvalue < alpha):
        print("We reject the null hypothesis")
else:
        print("We can't reject the null hypothesis")


# ## conclusion <a class="tocSkip">
# 
# > Company makes more revenue per user in 'ultimate' plan. Since, the average revenue obtained on ultimate plan is higher than in surf plan.
# Also, on testing hypothesis, that both mean are equal, it is rejected. So, it can be concluded that revenue per user made on 
# ultimate plan is higher than in surf plan.
# 
# > Likewise, the average revenue collection in NY-NJ area per user per month was less than in other areas. Also, in t test with null
# hypothesis _"NY-NJ area and other area have equal average revenue collection on averge per user"_   
# is rejected. So,the average revenue collection in 
# NY-NJ is less compared to other areas.

# <h2> Step 5. Overall Conclusion </h2>
# 
#     First data are downloaded.Checked for missing and duplicate data.
#     
#     Datatype of dates are changed and month is extracted and placed in another column.
#     
#     Call duration on every call are rounded up inorder to any possible error in calculation of rvenue.
#     
#     Pivot tables are created from each set of data so that further processing will be easy.
#     
#     data used are rounded up to get exact revenue while calculating.
#     
#     
#     Messages, data used and monthly call duration in both plan are calculated
#     
#                     messages   data_mb  monthly_call
#         plan                                      
#         surf         49014     26046956_     628084_
#         ultimate     27037     12394946_     288252_
#     
#     Mean value of call duration, messages sent and data used are determined.
#     
#     Revenue collected by each users are calculated.
#     
#     T-test is performed to check hypothesis.
#     
#     From hypothesis test it is obtained that:
#         
#     >More revenue  revenue ($ 70) is generated by ultimate plan user per month per user compared to surf plan users ($ 51).
#     
#     >Likewise, The NY-NJ area's user contibute less amount of average revenue ($ 53.31) per user per month compared to amount 
#     of revenue ($ 58.62) contibuted by users in other area
#     

# <div class="alert alert-success" role="alert">
# Reviewer's comment v. 3:
#     
# An excellent conclusion :)
# </div>

# In[ ]:




