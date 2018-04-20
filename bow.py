
# coding: utf-8

# In[1]:
print("Starting Python code.")

import pandas as pd
import numpy as np
import gzip
import os.path
import wget
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize


# In[2]:

# Download the dataset if the file doesn't exist on your machine
url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz'
if not os.path.isfile('reviews_Electronics_5.json.gz'):
    wget.download(url)
print("Data file exists now.")

# In[3]:

# Read the data from the file
df = pd.read_json('reviews_Electronics_5.json.gz', lines=True, compression='gzip')
print("DataFrame loaded.")

# In[5]:

# Drop unnecessary columns
df = df[['overall', 'reviewText']]
# Making the problem as a binary classification problem
df.loc[df['overall']>=4.0, 'label'] = 1
df.loc[df['overall']<4.0, 'label'] = 0

# Undersampling to deal with imbalanced data
neg_sample = df[df.label == 0]
pos_indices = df[df.label == 1].index
random_indices = np.random.choice(pos_indices, len(neg_sample), replace=False)
pos_sample = df.loc[random_indices]

df = pd.concat([pos_sample, neg_sample], ignore_index=True)
del(neg_sample, pos_sample)
print("Data shape after undersampling: ", df.shape)


# In[6]:

# Brief information about the new data
print(df['label'].value_counts()) # Now it's balanced
df.head()


# In[7]:

# Train/Evaluation/Test Dataset Split
X_trainval, X_test, y_trainval, y_test = train_test_split(np.asarray(df['reviewText']),
                                                   np.asarray(df['label']),
                                                   test_size = 0.2, random_state=777)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                             test_size = 0.2, random_state = 777)
del(df)


# In[8]:

# You might need to download the following packages from the downloader
# averaged_perceptron_tagger punkt stopwords wordnet

# nltk.download()


# In[9]:

# Lemmatization (consider this as an advanced version of stemming)
'''
Finds root words with linguistic rules
I borrowed and modified the code from https://www.kaggle.com/alvations/basic-nlp-with-nltk
preprocess_text function does:
tokenization for only English words with lemmatization after removing stopwords and punctuations and lower casing
'''

wnl = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english')+list(string.punctuation))
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text)
            if word not in stopwords_set
            and word.isalpha()]  # english words only (not punctuations nor numbers)


# In[10]:

# Example sentences with lemmatization and tokenization

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."
sent3 = "I am a high school student"

# Lemmatize and remove stopwords
processed_sent1 = preprocess_text(sent1)
processed_sent2 = preprocess_text(sent2)
processed_sent3 = preprocess_text(sent3)
print(processed_sent1)
print(processed_sent2)
print(processed_sent3)


# In[ ]:

# Vectorization
max_features=2000 # Use only the top 2000 frequent words as features
count_vect = CountVectorizer(max_features=max_features, 
                             analyzer=preprocess_text
                             ) 
X_train_counts = count_vect.fit_transform(X_train)
print("fit_transform completed.")
X_train_counts = X_train_counts.toarray()
X_val_counts = count_vect.transform(X_val).toarray()
X_test_counts = count_vect.transform(X_test).toarray()
print(count_vect.get_feature_names())
print(X_train_counts.shape)


# In[ ]:

# Used Naive Bayes as the baseline model
# With the parameter (Alpha) tuning
best_score=0
best_alpha=0
accs = []
for alpha in [1, 10, 100]:
    clf = MultinomialNB(alpha=alpha)
    NB = clf.fit(X_train_counts, y_train)
    acc = NB.score(X_val_counts, y_val)
    accs.append([acc, alpha])
    if acc > best_score:
        best_score = acc
        best_alpha = alpha
print(best_score, best_alpha)

clf = MultinomialNB(alpha=best_alpha)
NB = clf.fit(X_train_counts, y_train)
print("Test accuracy")
print(NB.score(X_test_counts, y_test))

'''
Without lemmantization, the test accuracy is 66.86% with NB.
'''


# In[ ]:

d_train = lgb.Dataset(X_train_counts, label=y_train)

params = {}
params['learning_rate'] = 0.2
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 1.0
params['num_leaves'] = 400
params['min_data'] = 100
params['min_data_in_bin'] = 4
params['min_data_in_leaf'] = 30
params['max_depth'] = -1
params['max_bin'] = 300
params['num_threads'] = 2
params['bagging_freq'] = 2
params['verbosity'] = 1

# Fitting
clf = lgb.train(params, d_train, num_boost_round = 100)

# Prediction
y_pred = clf.predict(X_test_counts)

for i in range(len(y_pred)):
    if y_pred[i]>=.5:       # setting threshold to .5
         y_pred[i]=1
    else:
         y_pred[i]=0

# Test score
print(accuracy_score(y_test, y_pred))

