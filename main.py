#!/usr/bin/env python

from flask import Flask, render_template, request
import pandas as pd
import pickle
import recipe2vec
import json
import os
import pandas as pd
import numpy as np
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



app = Flask(__name__)



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

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    
