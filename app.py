from flask import Flask, request, render_template
from model_ML import vectoriser, model

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    #vectorize
    X_test = vectoriser.transform([userText])    
    #Predict X_test
    result = model.predict_proba(X_test)
    prediction = result[0][0]

    #Compute the prediction to the response
    if prediction <= 0.4:
       response = ":)"
    elif prediction >= 0.6:
       response = ":("
    else : 
       response = ":|"
    
    return  response


if __name__ == "__main__":
    app.run()
