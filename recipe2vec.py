#!/bin/python
import os
import pandas as pd
import json
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from recipe_scrapers import scrape_me
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
lemmer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
from sklearn.metrics.pairwise import linear_kernel

def checkUrl(url1):
	sub_url1 = url1.split('/')
	
	url_title = []
	if sub_url1[2] == 'www.allrecipes.com' and sub_url1[3] == 'recipe' and RepresentsInt(sub_url1[4]) == True:
		ar_url = True
		url_title = sub_url1[5]
	else:
		ar_url = False
	
	if ar_url == False:
		if sub_url1[2] == 'www.epicurious.com' and sub_url1[3] == 'recipes' and sub_url1[4] == 'food' and sub_url1[5] == 'views':
			epi_url = True
			url_title = sub_url1[6]
		else:
			epi_url = False
	if ar_url == True:
		return('ar', url_title)
	elif epi_url == True:
		return('epi', url_title)
	else:
		return('None', url_title)

def RepresentsInt(s):
	try: 
		int(s)
		return True
	except ValueError:
		return False

def scrape_page(url1):
	
	d = {'ingredients': [], 'instructions': [], 'title': [], 'title_vegetarian': [], 'vegetarian': [], 'url': [], 'isVeg': []}
	page_scrape = scrape_me(url1)
	
	df = pd.DataFrame(d)
	
	df['ingredients'] = [page_scrape.ingredients()]
	df['instructions'] = page_scrape.instructions()
	df['title'] = page_scrape.title()
	df['url'] = url1
	
	
	# Grab the tagged list if it exists.
	if "egetarian" in df['title']:
		df['title_vegetarian'] = True
	else:
		df['title_vegetarian'] = False
	
	meats = ['ribs', 'offal', 'goat', 'duck', 'chicken', 'beef', 'pork', 
	 		'mutton', 'sausage', 'veal', 
	 		'pig', 'bacon', 'salami', 'prosciutto', 'tripe', 
	 		'turkey', 'liver', 'deer', 'rabbit','pot roast',
	 		'steak', 'steaks', 'pepperoni']
	
	df['vegetarian'] = True
	
	# Possible optimization
	for i, row in enumerate(df['ingredients'].astype(str)):
		for meat_item in meats:
			if meat_item in row.lower():
				df['vegetarian'][i] = False
				
	df['isVeg'] = False
	df.loc[(df['title_vegetarian'] == True),'isVeg']= True
	df.loc[(df['vegetarian'] == True),'isVeg'] = True   
	df['isVeg'] = (df['isVeg'] == True).astype(int)
	
	return(df)

def pre_process(text):
	# lowercase
	text=text.lower()
	# remove special characters and digits
	text=re.sub("(\\d|\\W)+"," ",text)
	
	words = text.split(' ')
	lem_words = [stemmer.stem(lemmer.lemmatize(word)) for word in words]
	sentence = ' '.join(lem_words)
	
	return sentence

def parts2cleanString(df_in, in_part, method):
	'''
	This function is designed to remove special characters and make lowercase for parts of a dataframe
	The input can be 'instructions', 'ingredients', or 'joint'
	The output of the function depends on the use-case or 'method', but is a dataframe column
	'''
	
	if method == 'tfidf':

		df_in['text'] = df_in['ingredients'].str.join(" ") + df_in['instructions']
		
		# Remove special characters, alpha-numeric digits, etc.
		df_in['text'] = df_in['text'].apply(lambda x:pre_process(x))   

		return(df_in['text'])

def RecipeSwap(df_in, method, ngram, parts):

	title_out = 'tmp'
	url_out = 'tmp'
	
	if method == 'tfidf':

		n = 0
		m = 1
		url1 = df_in['url'][0]
		title_in = df_in['title'][0]
		
		# Determine which part of the data to vectorize
		if parts == 'instructions':
			corpus = df_in['clean_instructions'].tolist()
		elif parts == 'ingredients':
			corpus = df_in['clean_ingredients'].tolist()
		else:
			corpus = df_in['text'].tolist()
		
		# Bring in stop words
		#from sklearn.feature_extraction import text
		#my_stop_words = text.ENGLISH_STOP_WORDS.union(corpus)
		#,stop_words=set(my_stop_words)
		
		tfidf_vec = TfidfVectorizer(max_df=0.5, ngram_range=(1,ngram))
		tfidf = tfidf_vec.fit_transform(corpus)
		cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
		related_docs_indices = cosine_similarities.argsort()[:-200:-1]
		
		for i in reversed(related_docs_indices[1:]):
			if df_in['isVeg'][i] == 1:
				title_out = df_in['title'][i]
				url_out = df_in['url'][i]
	
	return(title_in, title_out, url_out)
	
def RecipeSwap_with_tags(df_1, df_in, method, ngram, parts):

	title_out = 'tmp'
	url_out = 'tmp'
	
	if method == 'tfidf':
	
		n = 0
		m = 1
		url1 = df_1['url'][0]
		title_in = df_1['title'][0]
		dish_type = df_1['tag'][0]
		
		# Remove all non-veg dishes and other dish types
		df_out = df_in[(df_in['tag'] == dish_type) & (df_in['isVeg'] == 1)]
		
		# Might have to add line to remove any second cases of the same dish
		
		df = df_1.append(df_out)
		df.index = range(len(df))
		
		# Determine which part of the data to vectorize
		if parts == 'instructions':
			corpus = df['clean_instructions'].tolist()
		elif parts == 'ingredients':
			corpus = df['clean_ingredients'].tolist()
		elif parts == 'weighted_text':
		 	corpus = df['weighted_text'].tolist()
		else:
		 	corpus = df['text'].tolist()
		
		tfidf_vec = TfidfVectorizer(max_df=0.5, ngram_range=(1,ngram),stop_words=set(stopwords.words('english')+['advertis']))
		tfidf = tfidf_vec.fit_transform(corpus)
		
		cosine_similarities = linear_kernel(tfidf[0], tfidf).flatten()       
		related_docs_indices = cosine_similarities.argsort()[:-300:-1]
		
		title_out = df['title'][related_docs_indices[1]]
		url_out = df['url'][related_docs_indices[1]]
	
	return(title_in, title_out, url_out)



