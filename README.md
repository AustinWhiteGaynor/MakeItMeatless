# MakeItMeatless

Please visit
https://my-flask-image-kjsyrythxq-ue.a.run.app/

**Input:** Non-vegetarian recipe url from allrecipes.com or epicurious.com

**Output:** Vegetarian recipe recommendation and link

(Copy and paste any recipe from allrecipes.com or epicurious.com for a similar, but vegetarian, recommendation)

### *UNDER THE HOOD*

The input recipe will be scraped and passed through a trained **logistic regression** model that predicts the role in the meal that the recipe plays (dinner, dessert, etc). After determining the dish type, the combined text from the title and instructions is run through **TFIDF** calculations, with weights for verbs (spacy) from the instructions increased.

Following this transformation, the **cosine similarity** between the TFIDF vector and a corpus of recipes is used to return the most similar vegetarian recipe (still fitting the dish type).

Enjoy.
