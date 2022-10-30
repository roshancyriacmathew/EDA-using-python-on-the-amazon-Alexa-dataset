#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud


# In[2]:


df = pd.read_csv('amazon_alexa.tsv', sep='\t')
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


sns.countplot(x='rating', data=df)


# In[6]:


df['rating'].value_counts()


# In[7]:


fig = plt.figure(figsize=(7,7))
tags = df['rating'].value_counts()
tags.plot(kind='pie', autopct='%1.1f%%', label='')
plt.title("Distribution of the different ratings")
plt.show()


# In[8]:


fig = plt.figure(figsize=(20,10))
sns.countplot(y='variation', data=df)


# In[9]:


df.variation.value_counts().plot.barh(figsize=(12,7))
plt.title("Class distribution - Variations");


# In[10]:


df['variation'].value_counts()


# In[11]:


df[df['variation']=='Black  Dot']['rating'].value_counts()


# In[12]:


df[df['variation']=='Charcoal Fabric ']['rating'].value_counts()


# In[13]:


df[df['variation']=='Black  Dot']['feedback'].value_counts()


# In[14]:


sns.countplot(x='feedback', data=df)
plt.show()


# In[15]:


fig = plt.figure(figsize=(7,7))
tags = df['feedback'].value_counts()
tags.plot(kind='pie', autopct='%1.1f%%', label='')
plt.title("Distribution of the different sentiments")
plt.show()


# In[16]:


df['review_length'] = df.verified_reviews.str.len()
df.head()


# In[17]:


df['review_length'].describe()


# In[18]:


plt.hist(df['review_length'], bins=100)
plt.title("Histogram of review lengths")
plt.xlabel('Review Length')
plt.ylabel('Count')


# In[19]:


for i in range(5):
    print(df['verified_reviews'].iloc[i], "\n")
    print(df['feedback'].iloc[i], "\n")


# In[20]:


def data_processing(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+https\S+",'', text, flags = re.MULTILINE)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[21]:


df.verified_reviews = df['verified_reviews'].apply(data_processing)


# In[22]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[23]:


df['verified_reviews'] = df['verified_reviews'].apply(lambda x:stemming(x))


# In[24]:


for i in range(5):
    print(df['verified_reviews'].iloc[i], "\n")
    print(df['feedback'].iloc[i], "\n")


# In[25]:


pos_reviews = df[df.feedback==1]
pos_reviews.head()


# In[26]:


text = ' '.join([word for word in pos_reviews['verified_reviews']])
plt.figure(figsize=(20,15), facecolor=None)
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in positive reviews', fontsize=19)
plt.show()


# In[ ]:





# In[27]:


neg_reviews = df[df.feedback==0]
neg_reviews.head()


# In[28]:


text = ' '.join([word for word in neg_reviews['verified_reviews']])
plt.figure(figsize=(20,15), facecolor=None)
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in negative reviews', fontsize=19)
plt.show()


# In[ ]:




