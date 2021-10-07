import flask

import numpy as np
import pickle



app = flask.Flask(__name__, template_folder='templates')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print('Before load pickle')
    model = pickle.load(open('model/model_classifier.pkl', 'rb'))
    int_features = [int(x) for x in flask.request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = {0: 'not placed', 1: 'placed'}

    return flask.render_template('main.html', prediction_text='Student must be {} to workplace'.format(output[prediction[0]]))


@app.route('/')
def main():
    return(flask.render_template('main.html'))
if __name__ == "__main__":
    app.run(debug=True)