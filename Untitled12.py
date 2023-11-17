#!/usr/bin/env python
# coding: utf-8

# # MeriSKILL

# # Project 2 : Diabetes patients Analysis

# # the Attribute information:
# 1.Pregnancies: Number of times pregnant
# 
# 2.Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# 3.Blood pressure: Diastolic blood pressure (mm Hg)
# 
# 4.SkinThickness: Triceps skinfold thickness (mm)
# 
# 5.Insulin: 2-Hour serum insulin (mu U/ml) test
# 
# 6.BMI: Body mass index (weight in kg/(height in m)^2)
# 
# 7.DiabetesPedigreeFunction: A function that scores the likelihood of diabetes based on family history
# 
# 8.Age: Age in years
# 
# 9.Outcome: Class variable (0: the person is not diabetic or 1: the person is diabetic)

# # Reading the data

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


# load the Dataset
df = pd.read_csv(r"D:\intership\diabetes.csv")


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.head(10)


# # Variable Identification

# In[6]:


df.dtypes


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.nunique()


# In[10]:


# find the how many times the female hae been pregnent
df["Pregnancies"].unique()


# In[11]:


#1 patients are 17 times Pregnent
#135 patients are 1 times Pregnent
#111 patients are 0 times Pregnent
df["Pregnancies"].value_counts()


# In[12]:


# to check in percentage
df["Pregnancies"].value_counts()/len(df["Pregnancies"])


# In[13]:


df["BloodPressure"].unique()


# In[14]:


# 35 patient who have zero blood pressure
df["BloodPressure"].value_counts()


# In[15]:


# to check in percentage
df["BloodPressure"].value_counts()/len(df["BloodPressure"])


# In[16]:


# From the table
# The pregnancy numbers appear to be normally distributed whereas the others seem to be rightly skewed. 
# The mean and std deviation of pregnancies are more or less the same as opposed to the others
# Highest glucose levels is 199, pregnancies 17 and BMI 67
df.describe()


# # Data Visualization

# In[17]:


#Histrograms of each feature
df.hist(bins=10,figsize=(15,10))
plt.legend
plt.show()


# In[18]:


# Boxplot to detect outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient='h')
plt.title('Boxplot of Diabetes patient')
plt.show()


# In[19]:


#Close to 1 indicates a very good relationship
#Close to -1 indicates a very poor relationship
#From the above graph, we can see that the relationship is very strong for the below features
#Age-Pregnancies
#SkinThickness-BMI
#Glucose-Insulin

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap="Blues", annot=True)
plt.title('Correlation Diabetes Dataset')
plt.show()


# In[20]:


#we have observed in Pair plot analysis, we also found a strong relationship between age and pregnancy, Glucose Insulin
sns.pairplot(df, hue='Outcome')  # Pairplot for numeric variables, color by Outcome
plt.show()


# In[21]:


# Countplot for Pregnancies (diabetes) distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Pregnancies')
plt.title('Distribution of Pregnancies Outcome')
plt.show()


# In[22]:


#Countplot for BloodPressure (diabetes) distribution
plt.figure(figsize=(22, 8))
sns.countplot(data=df, x='BloodPressure')
plt.title('Distribution of BloodPressure Outcome')
plt.show()


# In[23]:


# Countplot for SkinThickness (diabetes) distribution
plt.figure(figsize=(22, 8))
sns.countplot(data=df, x='SkinThickness')
plt.title('Distribution of SkinThickness Outcome')
plt.show()


# In[24]:


# Understanding the number of women in different age groups with diabetes.
# all the women with diabetes most are from the age between 22 to 30.
#The frequency of women with diabetes decreases as age increases.
plt.hist(df[df['Outcome']==1]['Age'], bins = 5)
plt.title('Distribution of Age for Women who has Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[25]:


#Understanding the number of women in different age groups without diabetes.
#The highest number of Women without diabetes range between ages 22 to 33.
#Women between the age of 22 to 35 are at the highest risk of diabetes
#and also the is the highest number of those without diabetes.
plt.hist(df[df['Outcome']==0]['Age'], bins = 5)
plt.title('Distribution of Age for Women who do not have Diabetes')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[26]:


# Countplot for Outcome (diabetes) distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Outcome')
plt.title('Distribution of Diabetes Outcome')
plt.show()


# # We have seen that the behaviors of those with diabetes are around 35%, while those without diabetes are around 65%.
