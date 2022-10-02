```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```

# News Classification with Naive Bayes

we are going to perform the clasification of news dataset using the Naive Bayes Approach. For this project we have used the dataset from Kaggle https://www.kaggle.com/datasets/rmisra/news-category-dataset/. 

Before we get into the actual problem, we will broefly describe concept of  Bayes Theorem

### Bayes Theorem 

It describes the probability of an event, based on prior knowledge of conditions that might be related to the event. For example, if the risk of developing health problems is known to increase with age, Bayes' theorem allows the risk to an individual of a known age to be assessed more accurately (by conditioning it on their age) than simply assuming that the individual is typical of the population as a whole

![image.png](attachment:ce2c20e2-0ddf-4a45-9a02-dedd75440c5a.png)

the result P(A|B) is referred to as the posterior probability and P(A) is referred to as the prior probability.

P(A|B): Posterior probability.
P(A): Prior probability.


### Importing data 

we will import the data using Pandas library


```python
df = pd.read_json('data.json', lines = True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CRIME</td>
      <td>There Were 2 Mass Shootings In Texas Last Week...</td>
      <td>Melissa Jeltsen</td>
      <td>https://www.huffingtonpost.com/entry/texas-ama...</td>
      <td>She left her husband. He killed their children...</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENTERTAINMENT</td>
      <td>Will Smith Joins Diplo And Nicky Jam For The 2...</td>
      <td>Andy McDonald</td>
      <td>https://www.huffingtonpost.com/entry/will-smit...</td>
      <td>Of course it has a song.</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENTERTAINMENT</td>
      <td>Hugh Grant Marries For The First Time At Age 57</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/hugh-gran...</td>
      <td>The actor and his longtime girlfriend Anna Ebe...</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENTERTAINMENT</td>
      <td>Jim Carrey Blasts 'Castrato' Adam Schiff And D...</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/jim-carre...</td>
      <td>The actor gives Dems an ass-kicking for not fi...</td>
      <td>2018-05-26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENTERTAINMENT</td>
      <td>Julianna Margulies Uses Donald Trump Poop Bags...</td>
      <td>Ron Dicker</td>
      <td>https://www.huffingtonpost.com/entry/julianna-...</td>
      <td>The "Dietland" actress said using the bags is ...</td>
      <td>2018-05-26</td>
    </tr>
  </tbody>
</table>
</div>



### Exploratory Data Analysis

Lets Analyze the following features of the data 
- How many Datapoints are there.
- What are the each input features
- How many categories are there
- How many datapoints are there in each category

#### Total number of Data Points



```python
print("Size of Data : " , df.shape[0])
df.describe()
```

    Size of Data :  200853





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>category</th>
      <th>headline</th>
      <th>authors</th>
      <th>link</th>
      <th>short_description</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200853</td>
      <td>200853</td>
      <td>200853</td>
      <td>200853</td>
      <td>200853</td>
      <td>200853</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>41</td>
      <td>199344</td>
      <td>27993</td>
      <td>200812</td>
      <td>178353</td>
      <td>2309</td>
    </tr>
    <tr>
      <th>top</th>
      <td>POLITICS</td>
      <td>Sunday Roundup</td>
      <td></td>
      <td>https://www.huffingtonpost.comhttps://medium.c...</td>
      <td></td>
      <td>2013-01-17 00:00:00</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>32739</td>
      <td>90</td>
      <td>36620</td>
      <td>2</td>
      <td>19712</td>
      <td>100</td>
    </tr>
    <tr>
      <th>first</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012-01-28 00:00:00</td>
    </tr>
    <tr>
      <th>last</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-05-26 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>



#### Total Number of Categories
We will get the count of number of datapoints available for each of the categories. 


```python
# uniqueList  = list(df.category.unique())
# counts      = [0 for i in range(len(uniqueList))]
# histBucket =  dict(zip(uniqueList,counts))
# for i in df.category:
#     histBucket[i] += 1


# ### Sorted Histogram 
# sorted(histBucket.items(), key=lambda x: x[1],reverse=False)

# ## plot tht as a Histogram
# plt.bar(list(histBucket.keys())[:10],list( histBucket.values())[:10], color='g')
# plt.xticks(rotation=90)
# plt.title("Frequency of top 10 categories")

count = df['category'].value_counts()
plt.figure(figsize=(25,8))
sns.barplot(x=count.index,y=count.values)
plt.title("The distribution of categories")
plt.xlabel("Category")
plt.ylabel("The number of samples")

plt.xticks(rotation=60,fontsize = 14)
plt.show()
```


    
![png](output_7_0.png)
    


## Data Cleanup

Cleanup all the rows which has incomplete values in the dataframe and reduce the size of the data


```python
for i in df.columns:
    df[i].replace('',np.nan,inplace = True)
df.dropna( inplace=True)
print("Updated Values : " , df.shape)
count = df['category'].value_counts()
```

    Updated Values :  (148983, 6)


## Test Train Datasplit

Now we will split the data into the following
- Testing - 60%
- Development  - 20%
- Testing - 20%

After performing this, we need to perform a 5 fold cross validation with Respect to the given dataset. As of now, we will perform a 5 fold cross valdation with Testing and development dataset alone. 


```python

train, Test = \
              np.split(df.sample(frac=1, random_state=42), 
                       [int(.8*len(df))])

## now Split the Test data into test and development data for Cross Validation

train_kfold, dev_kfold = \
              np.split(train.sample(frac=1, random_state=42), 
                       [int(.8*len(train))])

print(" Testing Dataset : " , Test.shape[0])
print(" Training Dataset : " , train_kfold.shape[0])
print(" Dev Dataset : " , dev_kfold.shape[0])
```

     Testing Dataset :  29797
     Training Dataset :  95348
     Dev Dataset :  23838


### Cross Validation
Cross-validation is a resampling method that uses different portions of the data to test and train a model on different iterations. It is mainly used in settings where the goal is prediction, and one wants to estimate how accurately a predictive model will perform in practice

![image.png](attachment:124a315b-192f-4166-ba57-81c42c8aae6e.png)

## Get the Frequency count of all the words

we will now obtain the frequency of all the words in the given headlines and their description. This will be used to calculate all the probabilities of the given system

### Compute Prior Probability

First, We will compute the prior probability of all the classes available . this will compute the probability of occurence of each class in the given system 

The prior probability is the general probability of occurence of the class <br />
`P(Category)`  = $\frac{Num Of Occurences of category}{total data points}$  `   


```python
##  Computing prior probbiltiy
count = train_kfold['category'].value_counts()
priorProbability = {key: value for key, value in count.items()}
for k,v in count.items():
    priorProbability[k] = v/train_kfold.shape[0]
```

### Get all the count of words 

WE will now get all the count of words in the Headlines and the descriptions



```python
wordList = []
import re

mainDictionary = {}
for category,value in zip(train_kfold.category,train_kfold.headline):
    values = value.split(' ')
    values = list(map(str.lower,values))
    
    for k in values:
        ## Existing Values
        k = str(k)
        k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
        if( k_new not in mainDictionary ):
            mainDictionary[k_new] = {key: 0 for key, value in count.items()}
        
        mainDictionary[k_new][category] +=1;
```

#### Convert the dictionary into dataframe 


```python
freqWords = pd.DataFrame(mainDictionary)
import pickle

with open ('freqwords.dump', 'wb') as f:
    pickle.dump(freqWords,f)
```


```python
with open ('freqwords.dump', 'rb') as f:
    freqWords = pickle.load(f)
```

### Visualisation of the DataFrame for most frequency of Words

IN this section, we will visualise the most occurences of a word for a given category, based on the training data, This will ensure to get an idea on which are the most important keywords that would classsify a given news in certain category.




```python
# avoidList = ['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as','this','why','how','what','who','i']
avoidList = []
Location = freqWords.loc[:,~freqWords.columns.isin(avoidList)].idxmax(axis=1).to_list() 
Count    = freqWords.loc[:,~freqWords.columns.isin(avoidList)].max(axis=1).to_list()
category = freqWords.index.to_list()

mostUsedWords = pd.DataFrame([category,Location,Count], index=None)
mostUsedWords = mostUsedWords.transpose()
mostUsedWords

ax = sns.histplot(mostUsedWords[1])
ax.set_title("Histogram of TOp words in each category")

```




    Text(0.5, 1.0, 'Histogram of TOp words in each category')




    
![png](output_21_1.png)
    


## Elimination of Common words

Here we could observe that the most top word across all the categories was `the`. The reasons why we cannot use this as trainig data without preprocessing are

- The word `the` is very generic word, and we cannot let this word be the top word in every category as this will skew the probability 
- we need to eliminte these generalised words from our prediction such each category can have unique words which reperesent those categories

Upon analysis , we removed all the common words that were available in the code, and plotted the histogram to check for equal 
distribution

the words that were removed are 

```
['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as','this','why','how','what','who','i']
```



```python
avoidList = ['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as','this','why','how','what','who','i','are','that']
# avoidList = []
Location = freqWords.loc[:,~freqWords.columns.isin(avoidList)].idxmax(axis=1).to_list() 
Count    = freqWords.loc[:,~freqWords.columns.isin(avoidList)].max(axis=1).to_list()
category = freqWords.index.to_list()

mostUsedWords = pd.DataFrame([category,Location,Count], index=None)
mostUsedWords = mostUsedWords.transpose()
mostUsedWords

ax = sns.histplot(mostUsedWords[1])
ax.set_title("Histogram of TOp words in each category")

```




    Text(0.5, 1.0, 'Histogram of TOp words in each category')




    
![png](output_23_1.png)
    


#### Drop the Common words


Now we will remove the most common words from our dataframe in order to ignore them while computing the probabitity matrix




```python
freqWords = freqWords.drop(avoidList,axis=1)
freqWords.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boy</th>
      <th>iphone</th>
      <th>cases</th>
      <th>cute</th>
      <th>purses</th>
      <th>photos</th>
      <th>californians</th>
      <th>shouldnt</th>
      <th>gamble</th>
      <th>donald</th>
      <th>...</th>
      <th>mechanical</th>
      <th>mamahood</th>
      <th>stitched</th>
      <th>trumpcabinetband</th>
      <th>bummer</th>
      <th>chomskys</th>
      <th>mandera</th>
      <th>ringo</th>
      <th>through...</th>
      <th>angelous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>POLITICS</th>
      <td>14</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
      <td>27</td>
      <td>6</td>
      <td>37</td>
      <td>3</td>
      <td>1789</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>WELLNESS</th>
      <td>4</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>16</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ENTERTAINMENT</th>
      <td>24</td>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>230</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>TRAVEL</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>795</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>STYLE &amp; BEAUTY</th>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>16</td>
      <td>5</td>
      <td>1833</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49257 columns</p>
</div>



### Conditional Probability 

Now compute the conditional probability of the given data. i.e compute the probability of the occurance of `word` given its in a category `X`. 

` P( Word | Category ) `

Using this and the `P(Category)`, we should be able to get the posterior probabilities in term of Probability of a category given a word is present `P(Category|Word)`, which will be responsible for the classification of our data. 

In order to compute that, we will obtain the total number of occurences of every word on each class and append it as another column


```python
# freqWords["sum"] = freqWords.sum(axis=1)
ConditionalProb = freqWords.copy(deep=True)

ConditionalProb = ConditionalProb.div(ConditionalProb.sum(axis=1), axis=0)
ConditionalProb.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>boy</th>
      <th>iphone</th>
      <th>cases</th>
      <th>cute</th>
      <th>purses</th>
      <th>photos</th>
      <th>californians</th>
      <th>shouldnt</th>
      <th>gamble</th>
      <th>donald</th>
      <th>...</th>
      <th>mechanical</th>
      <th>mamahood</th>
      <th>stitched</th>
      <th>trumpcabinetband</th>
      <th>bummer</th>
      <th>chomskys</th>
      <th>mandera</th>
      <th>ringo</th>
      <th>through...</th>
      <th>angelous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>POLITICS</th>
      <td>0.000097</td>
      <td>0.000007</td>
      <td>0.000055</td>
      <td>0.000014</td>
      <td>0.000000</td>
      <td>0.000186</td>
      <td>0.000041</td>
      <td>0.000255</td>
      <td>0.000021</td>
      <td>0.012332</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>WELLNESS</th>
      <td>0.000086</td>
      <td>0.000021</td>
      <td>0.000385</td>
      <td>0.000021</td>
      <td>0.000000</td>
      <td>0.000342</td>
      <td>0.000000</td>
      <td>0.000150</td>
      <td>0.000000</td>
      <td>0.000021</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>ENTERTAINMENT</th>
      <td>0.000349</td>
      <td>0.000015</td>
      <td>0.000015</td>
      <td>0.000262</td>
      <td>0.000000</td>
      <td>0.000552</td>
      <td>0.000000</td>
      <td>0.000131</td>
      <td>0.000015</td>
      <td>0.003343</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>TRAVEL</th>
      <td>0.000032</td>
      <td>0.000032</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.025204</td>
      <td>0.000000</td>
      <td>0.000159</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>STYLE &amp; BEAUTY</th>
      <td>0.000127</td>
      <td>0.000153</td>
      <td>0.000153</td>
      <td>0.000407</td>
      <td>0.000127</td>
      <td>0.046612</td>
      <td>0.000000</td>
      <td>0.000102</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49257 columns</p>
</div>



Check whether the Sum of Probabilites add upto one for each category



```python
ConditionalProb.sum(axis=1).head()
PREDICTIONLABEL = ConditionalProb.index.to_list()
PREDICTIONLABEL_DICT = dict(zip(PREDICTIONLABEL,list(range(len(PREDICTIONLABEL))) ))
```

### NVB - Classifier. 

Get all the dev dataset and clean them, to get them ready for classification. 
- Seperate the actul value of the class 
- Seperate the words from the list and remove the most commonly used words





```python
yActual = dev_kfold.category.to_list()

Y_test = []
for line in dev_kfold.headline.to_list():
    lines = line.split(' ')
    values = list(map(str.lower,lines))
    newlist = [];
    for k in values:
        k = str(k);
        k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
        if(k_new not in avoidList):
            newlist.append(k_new)
    
    Y_test.append(newlist)
    
```

#### Prediction of Likelyhood
Now we will identify the probability of given headline for all the given classes,which will determine which class the given measure belongs to.


```python
Y_pred = []

for data in Y_test:
    probList = []
    for cat in PREDICTIONLABEL:
        probList.append(priorProbability[cat])
    probList = np.array(probList);
    for word in data:
        try:
            probList *= ConditionalProb[word].to_numpy()
        except KeyError:
            probList;
    Y_pred.append(probList)
```


```python
Y_pred_df = pd.DataFrame(Y_pred)

prevSize = Y_pred_df.shape[0];
print("Actual Size : " , prevSize)
##Remvove the entries with zero probability , to compute accuracy
Y_pred_df["idmax"] = Y_pred_df.idxmax(axis=1)
Y_pred_df["sum"] = Y_pred_df.sum(axis=1)


yPred_num = [];
yActual_num = [];
for i in range(len(yActual)):
#     if(Y_pred_df["sum"][i] > 0):
    yActual_num.append(PREDICTIONLABEL_DICT[yActual[i]] )
    yPred_num.append(Y_pred_df["idmax"][i])

from sklearn.metrics import accuracy_score, f1_score , precision_score

print("Accuracy Score : " , accuracy_score(yActual_num,yPred_num))
print("Precision Score : " , precision_score(yActual_num,yPred_num,average='macro'))
print("F1 Score " , f1_score(yActual_num,yPred_num,average='micro'))
```

    Actual Size :  23838
    Accuracy Score :  0.3937410856615488
    Precision Score :  0.35841167573919375
    F1 Score  0.3937410856615488


##  k Fold Cross Validation 

Now , we have coded the formulation to compute the accuracy on the given dataset, we will now do the same for k times on different dataset to compute the average accurcy of the model 


```python
from sklearn.metrics import accuracy_score, f1_score , precision_score
import re
def crossValidationNaiveBayes(train,kfold,PropConst):
    accuracyArray = []
    f1Array = []
    precisionArray = []
    for iterNum in range(kfold):
        print("Performing Cross Validation : " , iterNum)
        train_kfold, dev_kfold = \
                  np.split(train.sample(frac=1, random_state=np.random.randint(0,100)), 
                           [int(.8*len(train))])
        
        count = train_kfold['category'].value_counts()
        priorProbability = {key: value for key, value in count.items()}
        for k,v in count.items():
            priorProbability[k] = v/train_kfold.shape[0];
        
        wordList = []
        

        mainDictionary = {}
        for category,value in zip(train_kfold.category,train_kfold.headline):
            values = value.split(' ')
            values = list(map(str.lower,values))

            for k in values:
                ## Existing Values
                k = str(k)
                k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
                if( k_new not in mainDictionary ):
                    mainDictionary[k_new] = {key: 0 for key, value in count.items()}

                mainDictionary[k_new][category] +=1;
                
        freqWords = pd.DataFrame(mainDictionary)
        avoidList = ['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as','this','why','how','what','who','i','are','that']
        freqWords = freqWords.drop(avoidList,axis=1)
        ConditionalProb = freqWords.copy(deep=True)
        
        ConditionalProb = ConditionalProb.div(ConditionalProb.sum(axis=1), axis=0)
        
        yActual = dev_kfold.category.to_list()

        Y_test = []
        for line in dev_kfold.headline.to_list():
            lines = line.split(' ')
            values = list(map(str.lower,lines))
            newlist = [];
            for k in values:
                k = str(k);
                k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
                if(k_new not in avoidList):
                    newlist.append(k_new)

            Y_test.append(newlist)
        
        Y_pred = []
        
        for data in Y_test:
            probList = []
            for cat in PREDICTIONLABEL:
                probList.append(priorProbability[cat]*PropConst)
            probList = np.array(probList);
            for word in data:
                try:
                    probList *= ConditionalProb[word].to_numpy()
                except KeyError:
                    probList;
            Y_pred.append(probList)
        
        Y_pred_df = pd.DataFrame(Y_pred)

        prevSize = Y_pred_df.shape[0];
#         print("Actual Size : " , prevSize)
        ##Remvove the entries with zero probability , to compute accuracy
        Y_pred_df["idmax"] = Y_pred_df.idxmax(axis=1)
        Y_pred_df["sum"] = Y_pred_df.sum(axis=1)


        yPred_num = [];
        yActual_num = [];
        for i in range(len(yActual)):
        #     if(Y_pred_df["sum"][i] > 0):
            yActual_num.append(PREDICTIONLABEL_DICT[yActual[i]] )
            yPred_num.append(Y_pred_df["idmax"][i])

        accuracyArray.append(accuracy_score(yActual_num,yPred_num))
        precisionArray.append(precision_score(yActual_num,yPred_num,average='macro'))
        f1Array.append( f1_score(yActual_num,yPred_num,average='micro'))
        
        print("     Accuracy  : " , accuracyArray[iterNum])
        print("     Precision : " , precisionArray[iterNum])
        print("     F1        : " , f1Array[iterNum])
    
    return accuracyArray,precisionArray,f1Array
    
```


```python
accuracyArray,precisionArray,f1Array = crossValidationNaiveBayes(train,5,1000)
```

    Performing Cross Validation :  0
         Accuracy  :  0.3654249517576978
         Precision :  0.2348108306312544
         F1        :  0.3654249517576978
    Performing Cross Validation :  1
         Accuracy  :  0.3937410856615488
         Precision :  0.35841167573919375
         F1        :  0.3937410856615488
    Performing Cross Validation :  2
         Accuracy  :  0.36194311603322427
         Precision :  0.2281440311808766
         F1        :  0.36194311603322427
    Performing Cross Validation :  3
         Accuracy  :  0.38061078949576305
         Precision :  0.2803334823248514
         F1        :  0.38061078949576305
    Performing Cross Validation :  4
         Accuracy  :  0.3842604245322594
         Precision :  0.26403192144816384
         F1        :  0.38426042453225945



```python
print("Average Accuracy : " , sum(accuracyArray)/len(accuracyArray))
```

    Average Accuracy :  0.36582767010655254



```python
accuracyArray,precisionArray,f1Array = crossValidationNaiveBayes(train,5,10)
```

    Performing Cross Validation :  0
         Accuracy  :  0.3671448947059317
         Precision :  0.2729128742174741
         F1        :  0.3671448947059317
    Performing Cross Validation :  1
         Accuracy  :  0.36274016276533266
         Precision :  0.23055911911192603
         F1        :  0.36274016276533266
    Performing Cross Validation :  2
         Accuracy  :  0.3808624884637973
         Precision :  0.28391088401869774
         F1        :  0.3808624884637973
    Performing Cross Validation :  3
         Accuracy  :  0.3546438459602316
         Precision :  0.23314665975678162
         F1        :  0.3546438459602316
    Performing Cross Validation :  4
         Accuracy  :  0.3541823978521688
         Precision :  0.23004350751911895
         F1        :  0.3541823978521688



```python
print("Average Accuracy : " , sum(accuracyArray)/len(accuracyArray))
```

    Average Accuracy :  0.36391475794949235



```python
accuracyArray,precisionArray,f1Array = crossValidationNaiveBayes(count,train,5,100000000)
```

    Performing Cross Validation :  0
         Accuracy  :  0.36026512291299606
         Precision :  0.2546019861191974
         F1        :  0.360265122912996
    Performing Cross Validation :  1
         Accuracy  :  0.3811141874318315
         Precision :  0.26282274876770223
         F1        :  0.3811141874318315
    Performing Cross Validation :  2
         Accuracy  :  0.37087842939843946
         Precision :  0.25346032317025674
         F1        :  0.3708784293984395
    Performing Cross Validation :  3
         Accuracy  :  0.3688648376541656
         Precision :  0.2518455880192121
         F1        :  0.3688648376541656
    Performing Cross Validation :  4
         Accuracy  :  0.3546438459602316
         Precision :  0.23314665975678162
         F1        :  0.3546438459602316


### Smoothing Operation 

Laplace smoothing is a smoothing technique that helps tackle the problem of zero probability in the Naïve Bayes machine learning algorithm

It modififies the computation of probabilities by making the following change to the formula 


$P(word| category) =  \frac{Num of Occurances of word in that category  + \alpha}{N*\alpha + numOf Occurences of word in total}$

Here $\alpha$ is the smoothing parameter, which is generally taken as 1 


```python
from sklearn.metrics import accuracy_score, f1_score , precision_score
import re
def crossValidationNaiveBayesWithSmoothing(count,train,kfold,PropConst,alpha):
    accuracyArray = []
    f1Array = []
    precisionArray = []
    for iterNum in range(kfold):
        print("Performing Cross Validation : " , iterNum)
        train_kfold, dev_kfold = \
                  np.split(train.sample(frac=1, random_state=np.random.randint(0,100)), 
                           [int(.8*len(train))])
        
        count = train_kfold['category'].value_counts()
        priorProbability = {key: value for key, value in count.items()}
        for k,v in count.items():
            priorProbability[k] = v/train_kfold.shape[0];
        
        wordList = []
        

        mainDictionary = {}
        for category,value in zip(train_kfold.category,train_kfold.headline):
            values = value.split(' ')
            values = list(map(str.lower,values))

            for k in values:
                ## Existing Values
                k = str(k)
                k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
                if( k_new not in mainDictionary ):
                    mainDictionary[k_new] = {key: 0 for key, value in count.items()}

                mainDictionary[k_new][category] +=1;
                
        freqWords = pd.DataFrame(mainDictionary)
        avoidList = ['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as','this','why','how','what','who','i','are','that']
        freqWords = freqWords.drop(avoidList,axis=1)
        freqWords += alpha;
        ConditionalProb = freqWords.copy(deep=True)
        
        
        ConditionalProb = ConditionalProb.div(ConditionalProb.sum(axis=1) + len(PREDICTIONLABEL)*alpha, axis=0)
        
        yActual = dev_kfold.category.to_list()

        Y_test = []
        for line in dev_kfold.headline.to_list():
            lines = line.split(' ')
            values = list(map(str.lower,lines))
            newlist = [];
            for k in values:
                k = str(k);
                k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
                if(k_new not in avoidList):
                    newlist.append(k_new)

            Y_test.append(newlist)
        
        Y_pred = []
        
        for data in Y_test:
            probList = []
            for cat in PREDICTIONLABEL:
                probList.append(priorProbability[cat]*PropConst)
            probList = np.array(probList);
            for word in data:
                try:
                    probList *= ConditionalProb[word].to_numpy()
                except KeyError:
                    probList *= np.ones_like(ConditionalProb.iloc[:,0])*(alpha/len(PREDICTIONLABEL));
            Y_pred.append(probList)
        
        Y_pred_df = pd.DataFrame(Y_pred)

        prevSize = Y_pred_df.shape[0];
#         print("Actual Size : " , prevSize)
        ##Remvove the entries with zero probability , to compute accuracy
        Y_pred_df["idmax"] = Y_pred_df.idxmax(axis=1)
        Y_pred_df["sum"] = Y_pred_df.sum(axis=1)


        yPred_num = [];
        yActual_num = [];
        for i in range(len(yActual)):
        #     if(Y_pred_df["sum"][i] > 0):
            yActual_num.append(PREDICTIONLABEL_DICT[yActual[i]] )
            yPred_num.append(Y_pred_df["idmax"][i])

        accuracyArray.append(accuracy_score(yActual_num,yPred_num))
#         precisionArray.append(precision_score(yActual_num,yPred_num,average='macro'))
#         f1Array.append( f1_score(yActual_num,yPred_num,average='micro'))
        
        print("     Accuracy  : " , accuracyArray[iterNum])
#         print("     Precision : " , precisionArray[iterNum])
#         print("     F1        : " , f1Array[iterNum])
    
    return accuracyArray
    
```


```python
crossValidationNaiveBayesWithSmoothing(count,train,5,10,1)
```

    Performing Cross Validation :  0
         Accuracy  :  0.4813323265374612
    Performing Cross Validation :  1
         Accuracy  :  0.45163184830942193
    Performing Cross Validation :  2
         Accuracy  :  0.44835976172497694
    Performing Cross Validation :  3
         Accuracy  :  0.470970719020052
    Performing Cross Validation :  4
         Accuracy  :  0.4700897726319322





    [0.4813323265374612,
     0.45163184830942193,
     0.44835976172497694,
     0.470970719020052,
     0.4700897726319322]




```python
crossValidationNaiveBayesWithSmoothing(count,train,5,10,2)
```

    Performing Cross Validation :  0
         Accuracy  :  0.42201526973739406
    Performing Cross Validation :  1
         Accuracy  :  0.40464804094303214
    Performing Cross Validation :  2
         Accuracy  :  0.4380401040355735
    Performing Cross Validation :  3
         Accuracy  :  0.4399697961238359
    Performing Cross Validation :  4
         Accuracy  :  0.4351455659031798





    [0.42201526973739406,
     0.40464804094303214,
     0.4380401040355735,
     0.4399697961238359,
     0.4351455659031798]




```python
crossValidationNaiveBayesWithSmoothing(count,train,5,10,5)
```

    Performing Cross Validation :  0
         Accuracy  :  0.37498951254299856
    Performing Cross Validation :  1
         Accuracy  :  0.37649970635120394
    Performing Cross Validation :  2
         Accuracy  :  0.38400872556422516
    Performing Cross Validation :  3
         Accuracy  :  0.38916855440892695
    Performing Cross Validation :  4
         Accuracy  :  0.3800654417316889





    [0.37498951254299856,
     0.37649970635120394,
     0.38400872556422516,
     0.38916855440892695,
     0.3800654417316889]




```python
crossValidationNaiveBayesWithSmoothing(count,train,5,10,10)
```

    Performing Cross Validation :  0
         Accuracy  :  0.34528903431495933
    Performing Cross Validation :  1
         Accuracy  :  0.3381575635539894
    Performing Cross Validation :  2
         Accuracy  :  0.33517912576558434
    Performing Cross Validation :  3
         Accuracy  :  0.3381575635539894
    Performing Cross Validation :  4
         Accuracy  :  0.3431915429146741





    [0.34528903431495933,
     0.3381575635539894,
     0.33517912576558434,
     0.3381575635539894,
     0.3431915429146741]




```python
crossValidationNaiveBayesWithSmoothing(count,train,5,10,20)
```

    Performing Cross Validation :  0
         Accuracy  :  0.3141622619347261
    Performing Cross Validation :  1
         Accuracy  :  0.3177699471432167
    Performing Cross Validation :  2
         Accuracy  :  0.31298766675056633
    Performing Cross Validation :  3
         Accuracy  :  0.3126520681265207
    Performing Cross Validation :  4
         Accuracy  :  0.3103867774142126





    [0.3141622619347261,
     0.3177699471432167,
     0.31298766675056633,
     0.3126520681265207,
     0.3103867774142126]




```python
crossValidationNaiveBayesWithSmoothing(count,train,5,10,0.8)
```

    Performing Cross Validation :  0
         Accuracy  :  0.458595519758369
    Performing Cross Validation :  1
         Accuracy  :  0.48972229213860224
    Performing Cross Validation :  2
         Accuracy  :  0.4333836731269402
    Performing Cross Validation :  3
         Accuracy  :  0.46090276029868277
    Performing Cross Validation :  4
         Accuracy  :  0.456456078530078





    [0.458595519758369,
     0.48972229213860224,
     0.4333836731269402,
     0.46090276029868277,
     0.456456078530078]



### Validation on the testing set 

Now we perform the validation of our model on the final testing set, that we had set aside at the begining and test our accuracy. 


```python
def crossValidationNaiveBayesWithSmoothing_test(count,train,PropConst,alpha):
    train_kfold =  train
    count = train_kfold['category'].value_counts()
    priorProbability = {key: value for key, value in count.items()}
    for k,v in count.items():
        priorProbability[k] = v/train_kfold.shape[0];

    wordList = []


    mainDictionary = {}
    for category,value in zip(train_kfold.category,train_kfold.headline):
        values = value.split(' ')
        values = list(map(str.lower,values))

        for k in values:
            ## Existing Values
            k = str(k)
            k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
            if( k_new not in mainDictionary ):
                mainDictionary[k_new] = {key: 0 for key, value in count.items()}

            mainDictionary[k_new][category] +=1;

    freqWords = pd.DataFrame(mainDictionary)
    avoidList = ['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as','this','why','how','what','who','i','are','that']
    freqWords = freqWords.drop(avoidList,axis=1)
    freqWords += alpha;
    ConditionalProb = freqWords.copy(deep=True)


    ConditionalProb = ConditionalProb.div(ConditionalProb.sum(axis=1) + len(PREDICTIONLABEL)*alpha, axis=0)

    yActual = dev_kfold.category.to_list()

    Y_test = []
    for line in dev_kfold.headline.to_list():
        lines = line.split(' ')
        values = list(map(str.lower,lines))
        newlist = [];
        for k in values:
            k = str(k);
            k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
            if(k_new not in avoidList):
                newlist.append(k_new)

        Y_test.append(newlist)

    Y_pred = []
    
    for data in Y_test:
        probList = []
        for cat in PREDICTIONLABEL:
            probList.append(priorProbability[cat]*PropConst)
        probList = np.array(probList);
        for word in data:
            try:
                probList *= ConditionalProb[word].to_numpy()
            except KeyError:
                probList *= np.ones_like(ConditionalProb.iloc[:,0])*(alpha/len(PREDICTIONLABEL));
        Y_pred.append(probList)

    Y_pred_df = pd.DataFrame(Y_pred)

    prevSize = Y_pred_df.shape[0];
#         print("Actual Size : " , prevSize)
    ##Remvove the entries with zero probability , to compute accuracy
    Y_pred_df["idmax"] = Y_pred_df.idxmax(axis=1)
    Y_pred_df["sum"] = Y_pred_df.sum(axis=1)


    yPred_num = [];
    yActual_num = [];
    for i in range(len(yActual)):
    #     if(Y_pred_df["sum"][i] > 0):
        yActual_num.append(PREDICTIONLABEL_DICT[yActual[i]] )
        yPred_num.append(Y_pred_df["idmax"][i])


    print("     Accuracy  : " ,accuracy_score(yActual_num,yPred_num))
#         print("     Precision : " , precisionArray[iterNum])
#         print("     F1        : " , f1Array[iterNum])

    return accuracy
```


```python
accuracy = crossValidationNaiveBayesWithSmoothing_test(count,Test,10,1)
print(accuracy)
```

         Accuracy  :  0.4051514388791006
    [0.36026512291299606, 0.3811141874318315, 0.37087842939843946, 0.3688648376541656, 0.3546438459602316]


### Top words on Each category 

Now we are intrested in finding top words in each category


```python
wordList = []
import re

count = train['category'].value_counts()
mainDictionary = {}
for category,value in zip(train.category,train.headline):
    values = value.split(' ')
    values = list(map(str.lower,values))
    
    for k in values:
        ## Existing Values
        k = str(k)
        k_new = re.sub('[^a-zA-Z0-9 \n\.]', '', k)
        if( k_new not in mainDictionary ):
            mainDictionary[k_new] = {key: 0 for key, value in count.items()}
        
        mainDictionary[k_new][category] +=1;

freqWords = pd.DataFrame(mainDictionary)
avoidList = ['the', 'to', 'a','of','in','and','for','is','on','your','with','my','you','at','as',\
             'after','her','may','this','why','how','what','who','i','are','that','be','it','from']
freqWords = freqWords.drop(avoidList,axis=1)
freqWords += 1;
ConditionalProb = freqWords.copy(deep=True)


ConditionalProb = ConditionalProb.div(ConditionalProb.sum(axis=1) + len(PREDICTIONLABEL)*1, axis=0)

```


```python
nlargest = 10
order = np.argsort(-ConditionalProb.values, axis=1)[:, :nlargest]
result = pd.DataFrame(ConditionalProb.columns[order], 
                      columns=['top{}'.format(i) for i in range(1, nlargest+1)],
                      index=ConditionalProb.index)

print(result)
```

                         top1       top2        top3        top4        top5  \
    POLITICS            trump     donald      trumps         gop       about   
    WELLNESS             life     health          do         can         new   
    ENTERTAINMENT         new      about       trump         his        star   
    TRAVEL             photos     travel        best          10         new   
    STYLE & BEAUTY     photos    fashion       style        week               
    PARENTING            kids    parents       about    children         mom   
    HEALTHY LIVING     health      about         can        life           5   
    QUEER VOICES          gay        new       queer       about       trans   
    FOOD & DRINK      recipes       best        food        make      photos   
    BUSINESS         business      women         new                   about   
    COMEDY              trump     donald     colbert      trumps     stephen   
    PARENTS              kids        mom       about     parents        moms   
    SPORTS                nfl        his        team     olympic         nba   
    HOME & LIVING      photos       home       video       ideas         day   
    BLACK VOICES        black        new      police       about       white   
    IMPACT                           day       world        help         new   
    WOMEN               women      about       woman      sexual      womens   
    THE WORLDPOST        u.s.      trump      attack        says       syria   
    MEDIA               trump       news      donald         fox         new   
    CRIME              police        man    shooting        cops   allegedly   
    WEIRD NEWS            man        out        into         his       watch   
    TASTE             recipes       make        food         new          10   
    WORLD NEWS          north      korea       trump        u.s.        says   
    RELIGION             pope    francis  meditation       daily      church   
    DIVORCE           divorce               divorced    marriage        when   
    GREEN             climate     change              california         new   
    TECH                apple        new    facebook      iphone      google   
    WEDDINGS          wedding   marriage           5         day      photos   
    STYLE             fashion        new      beauty       style        best   
    MONEY              credit      money         tax   financial        more   
    SCIENCE             video      space         new     science  scientists   
    ARTS & CULTURE        art        new        book       trump      artist   
    WORLDPOST             war        not       world                     new   
    FIFTY               about       when        life           5               
    GOOD NEWS             his        dog         man    homeless         boy   
    LATINO VOICES      latino      trump     latinos       about      puerto   
    EDUCATION          school  education     schools    teachers    students   
    ARTS                  art                  first     nighter          an   
    COLLEGE           college   students  university     student      campus   
    ENVIRONMENT       climate     change       video         new         oil   
    CULTURE & ARTS  imageblog     photos         art               interview   
    
                        top6       top7      top8         top9     top10  
    POLITICS         clinton        his      says      hillary       new  
    WELLNESS                     cancer         5           we     about  
    ENTERTAINMENT       says       will       out        first    donald  
    TRAVEL                 5      world    hotels         most            
    STYLE & BEAUTY      more      photo       new        video    beauty  
    PARENTING           baby  parenting       day         when     child  
    HEALTHY LIVING      more       ways      when          new    people  
    QUEER VOICES       lgbtq        out      lgbt  transgender  marriage  
    FOOD & DRINK                     10         5       recipe       new  
    BUSINESS             its       wall      more          can       ceo  
    COMEDY             jimmy       bill       his          snl     about  
    PARENTS             baby  parenting      when          day     their  
    SPORTS          football       game     super       player      u.s.  
    HOME & LIVING      house        diy     craft         make       new  
    BLACK VOICES        says     photos       his        women        we  
    IMPACT               can        our  homeless        women        we  
    WOMEN                men      trump    tweets          not       day  
    THE WORLDPOST      china   election      isis     refugees     north  
    MEDIA              media       over      says        about    trumps  
    CRIME             killed      found     video         dead   suspect  
    WEIRD NEWS           dog        its     woman         just     trump  
    TASTE               best          5      will          day     these  
    WORLD NEWS           new        its    crisis        south      over  
    RELIGION          muslim  religious        an        about       not  
    DIVORCE               do        not         5       should        ex  
    GREEN                 we        our       not          its      more  
    TECH                week       will     watch          can      tech  
    WEDDINGS            tips         10   married         ways     bride  
    STYLE                all      dress      week       makeup     these  
    MONEY               ways        can     about          not       new  
    SCIENCE            study       nasa      mars        earth     shows  
    ARTS & CULTURE     about      women   artists           by       his  
    WORLDPOST           will       u.s.      iran       israel       can  
    FIFTY                 50       best    things   retirement       day  
    GOOD NEWS             by       teen      help          out     their  
    LATINO VOICES        new       says     video       trumps   america  
    EDUCATION                   teacher   student        about    public  
    ARTS                 new      stage   theatre         door      film  
    COLLEGE           sexual      about  colleges    education   assault  
    ENVIRONMENT           by  hurricane     spill     keystone     sandy  
    CULTURE & ARTS        by        new    artist        video   gallery  


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:3: DeprecationWarning: Support for multi-dimensional indexing (e.g. `index[:, None]`) on an Index is deprecated and will be removed in a future version.  Convert to a numpy array before indexing instead.
      This is separate from the ipykernel package so we can avoid doing imports until



```python

```
