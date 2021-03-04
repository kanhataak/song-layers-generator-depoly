# from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import Flask,render_template,url_for,request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.models import  Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,LSTM ,Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
import pandas as pd

model = load_model('song_lyrics_generator.h5')

def complete_this_song(seed_texts,next_words):
    for _ in range(next_words):
        # Tokenization
        df = pd.read_csv("final_song_df.csv")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df['Lyric'].astype(str).str.lower())



        # df = pd.read_csv("final_song_df.csv")
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(df['Lyric'].astype(str).str.lower())
        token_list = tokenizer.texts_to_sequences([seed_texts])[0]
        token_list = pad_sequences([token_list], maxlen=655, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_texts += " " + output_word
    return seed_texts

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():
        seed_texts = request.form['text']
        next_words = int(request.form['words'])
        seed_text = complete_this_song(seed_texts,next_words)
        # return jsonify({"Song layers: {}".format(seed_text)})
        # return jsonify(f'Song layers:',seed_texts)
        return render_template('index.html', prediction_text ='Song layers: {}'.format(seed_text))


if __name__ == '__main__':
	app.run(debug=True)
