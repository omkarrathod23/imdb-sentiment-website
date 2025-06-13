from flask import Flask, render_template, request
import pickle
import numpy as np
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences
from preprocess import clean_text

app = Flask(__name__)

# Load model and tokenizer
model = load_model("sentiment_model.h5")
with open("tokenizer.pickle", "rb") as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    review = request.form.get("review", "")

    cleaned = clean_text(review)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)

    prediction = float(model.predict(padded)[0][0])
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    positive_conf = round(prediction * 100, 2)
    negative_conf = round((1 - prediction) * 100, 2)

    # Pass raw values for chart, rounded for display
    return render_template(
        "result.html",
        review=review,
        sentiment=sentiment,
        positive=positive_conf,
        negative=negative_conf,
        positive_raw=prediction * 100,
        negative_raw=(1 - prediction) * 100
    )

if __name__ == '__main__':
    app.run(debug=True)
