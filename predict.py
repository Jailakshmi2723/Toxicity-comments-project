import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessing import clean_text

# Load trained model
model = load_model("models/toxicity_model.keras")

# Load tokenizer
with open("models/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)


def predict_toxicity(text):
    # Clean text
    cleaned_text = clean_text(text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    # Predict
    prediction = model.predict(padded_sequence)

    return prediction