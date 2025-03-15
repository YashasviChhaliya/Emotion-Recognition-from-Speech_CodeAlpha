# Emotion-Recognition-from-Speech_CodeAlpha

🎙️ Speech Emotion Recognition Model - Explanation
📌 Overview
This model analyzes speech audio files and predicts the emotion expressed in the speech. It is a deep learning-based classification model trained on the SAVEE dataset. The model extracts important features from speech, such as MFCCs, Chroma, and Mel Spectrograms, and uses them to classify speech into different emotions.

⚙️ Model Architecture
The model is a Deep Neural Network (DNN) built using TensorFlow/Keras with the following architecture:

1️⃣ Input Layer → Accepts the extracted audio features
2️⃣ Hidden Layer 1 → 256 neurons, ReLU activation
3️⃣ Dropout Layer → Prevents overfitting (30% dropout)
4️⃣ Hidden Layer 2 → 128 neurons, ReLU activation
5️⃣ Dropout Layer → Prevents overfitting
6️⃣ Hidden Layer 3 → 64 neurons, ReLU activation
7️⃣ Output Layer → Softmax activation (7 classes, one for each emotion)

📌 Model Code:

python
Copy
Edit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    Dropout(0.3),  # Dropout for regularization
    Dense(128, activation='relu'),  # Hidden Layer 1
    Dropout(0.3),
    Dense(64, activation='relu'),  # Hidden Layer 2
    Dense(len(np.unique(y)), activation='softmax')  # Output Layer (7 emotions)
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
🎤 How the Model Works
1️⃣ Feature Extraction

Converts audio files into numerical features using MFCCs, Chroma, and Mel Spectrograms.
2️⃣ Deep Learning Classification

The extracted features are fed into a Deep Neural Network (DNN) for classification.
3️⃣ Prediction

The model predicts one of seven emotions:
Angry 😡
Disgust 🤢
Fearful 😨
Happy 😀
Neutral 😐
Sad 😢
Surprised 😲
📊 Model Performance
The model is trained for 50 epochs and achieves high accuracy (>80%) on the SAVEE dataset.
To improve performance, we use dropout layers to prevent overfitting.
🚀 How the Model is Used
✔ In Flask API:

The trained model is loaded in Flask (app.py), which accepts audio files and predicts emotions.
✔ For Real-time Applications:
Can be integrated into chatbots, call centers, mental health monitoring apps, etc.
🎯 Future Improvements
Use CNNs/RNNs instead of a simple DNN for better accuracy.
Train on a larger dataset for improved generalization.
Implement real-time speech emotion detection using a microphone.
