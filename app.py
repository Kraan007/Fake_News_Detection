from flask import Flask, request,render_template
import pickle


model = pickle.load(open("finalized_model.pkl","rb"))
vector = pickle.load(open("vectorizer.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        news = request.form['news']
        prediction = model.predict(vector.transform([news]))[0]
        print(prediction)

        return render_template("prediction.html",prediction_text = f"News headline is - {prediction}")
    else:
        return render_template('prediction.html')


if __name__ == "__main__":
    app.run(debug=True)