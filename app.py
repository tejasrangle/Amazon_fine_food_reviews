from flask import *
import pickle

import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


from bs4 import BeautifulSoup

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# def transform(review):
#     sentance = re.sub(r"http\S+", "", sentance)
#     sentance = BeautifulSoup(sentance, 'lxml').get_review()
#     sentance = re.sub("\S*\d\S*", "", sentance).strip()
#     sentance = re.sub('[^A-Za-z]+', ' ', sentance)
#     sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
#     return sentance.strip()

def transform(review):
    review = review.lower()
    review = nltk.word_tokenize(review)

    y = []
    for i in review:
        if i.isalnum():
            y.append(i)

    review = y[:]
    y.clear()

    for i in review:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    review = y[:]
    y.clear()

    for i in review:
        y.append(ps.stem(i))

    return " ".join(y)

def result(tr_review):
    vector_input = tfidf.transform([tr_review])
    result = model.predict(vector_input)[0]
    return result





app = Flask(__name__)

@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/action",methods=["POST"])
def action():
    review=request.form["review"]
    tr_review=transform(review)
    prediction=result(tr_review)
    return render_template("show.html",result=prediction)


if __name__ =="__main__":
    app.run(debug=True)