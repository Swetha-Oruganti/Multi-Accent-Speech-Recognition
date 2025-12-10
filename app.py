from fastapi import FastAPI, UploadFile, File
import numpy as np
import librosa
import tensorflow as tf

app = FastAPI()

# Load model
model = tf.keras.models.load_model("your_model.keras")  # change if different name

@app.get("/")
def home():
    return {"message": "Speech Accent Recognition API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load audio
    audio, sr = librosa.load(file.file, sr=22050)

    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Pad / reshape to your input size (1172 x 13)
    max_length = 1172
    if mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0, pad_width)), mode='constant')

    mfcc = mfcc[:, :max_length]
    mfcc = mfcc.T  # (1172, 13)
    mfcc = np.expand_dims(mfcc, axis=0)

    # Predict
    pred = model.predict(mfcc)
    predicted_class = int(np.argmax(pred))

    return {"predicted_class": predicted_class}
