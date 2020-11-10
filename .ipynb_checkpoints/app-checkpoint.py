# Importing libraries

import numpy as np
from flask import Flask, request, render_template
from flask_cors import CORS
import os
from sklearn.externals import joblib
import pickle
import flask
import newspaper
from newspaper import Article
import urllib

# Loading Flask and assigning the model variable

app = Flask(__name__)
CORS(app)
app = Flask(__name__, template_folder="templates")

with open('model.pickle', 'rb') as handle:
    model = pickle.load(handle)
    
@app.route('/')
def main():
    return render_template('new.html')

# Receiving the input url from the user and using Web Scraping to extract the news content
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)
    print(url)
    url = urllib.parse.unquote(url[5:])
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = article.summary
    
    # Passing the news and predict if it's Fake or Real
    pred = model.predict([news])
    pred_text = 'FAKE' if pred[0] == 0 else 'REAL'
    return render_template('new.html', prediction_text='The news is "{}"'.format(pred_text))

if __name__=="__main__":
    port=int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)