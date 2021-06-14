from flask import Flask
from flask import render_template, url_for, request

app = Flask(__name__)

import pickle as pk

model='FakeNewsModel.pkl'
transform = 'transform.pkl'
clf = pk.load(open(model, 'rb'))
cv = pk.load(open(transform, 'rb'))

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  if request.method=='POST':
    txt = request.form['txt']
    data = [txt]
    vector = cv.transform(data).toarray()
    predicting = clf.predict(vector)
  return render_template('result.html', prediction=predicting)
if __name__ == '__main__':
  app.run(debug=True)