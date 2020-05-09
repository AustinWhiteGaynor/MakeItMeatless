""" This file largely follows the steps outlined in the Insight Flask tutorial, except data is stored in a
flat csv (./assets/births2012_downsampled.csv) vs. a postgres database. If you have a large database, or
want to build experience working with SQL databases, you should refer to the Flask tutorial for instructions on how to
query a SQL database from here instead.

May 2019, Donald Lee-Brown
"""

from flask import render_template
from flaskexample import app
from flaskexample.a_model import ModelIt
import pandas as pd
from flask import request
import pickle

import recipe2vec

import pandas as pd
import json
import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from recipe_scrapers import scrape_me
from nltk.corpus import stopwords
lemmer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()
from sklearn.metrics.pairwise import linear_kernel

# here's the homepage
@app.route('/')
def homepage():
	blah = 'main_page'
	return render_template("index.html", var=blah)

@app.route('/result', methods=['POST', 'GET'])
def product():
	if request.method == 'POST':
		result = request.form

		url1=result['Name']

		[page, url_title] = recipe2vec.checkUrl(url1)

		if page == 'ar' or page == 'epi':
			
			df_new = recipe2vec.scrape_page(url1)

			df_new['text'] = recipe2vec.parts2cleanString(df_new, 'both', 'tfidf')
			#print(df_new['text'])
			df_ar = pd.read_json('ar_tagged_and_clean.json', orient='columns')
			df_ar.index = range(len(df_ar))
			
			# load the model from disk
			classifier = pickle.load(open('classifier.sav', 'rb'))
			vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
			# Use the log. regression model
			y_predict = classifier.predict(vectorizer.transform(df_new['text']))
			df_new['tag'] = y_predict

			[title1, title2, url2] = recipe2vec.RecipeSwap_with_tags(df_new, df_ar, 'tfidf', 2, 'both')

		else:
			title1 = 'error'
			title2 = 'error'
			url2 = 'url not a good recipe'
		return render_template("result.html", url1 = url1, title1 = title1, title2 = title2, url2 = url2)

# now let's do something fancier - take an input, run it through a model, and display the output on a separate page

