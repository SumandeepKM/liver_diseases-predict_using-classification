from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('Random_forest.pkl', 'rb'))
@app.route('/')
def Home():
    return render_template('index.html')



@app.route("/predict", methods=['POST'])
def predict():
    
    
      
   int_features = [int(float(x)) for x in request.form.values()]
   final_features = [np.array(int_features)]
   prediction = model.predict(final_features)
   
   output = prediction
   
   return render_template('index.html', prediction_text = 'liver is  $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)


