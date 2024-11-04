# Speech Emotion Recognition Using LSTM

## Overview

This project involves building a Speech Emotion Recognition (SER) model using Long Short-Term Memory (LSTM) neural networks to identify emotions conveyed in speech. The model processes audio signals to detect emotions such as happiness, sadness, anger, and neutrality. Applications of this project include improving user experience in virtual assistants, enhancing sentiment analysis in customer service, and supporting mental health monitoring.

## Dataset

Popular datasets for SER include:

- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- TESS (Toronto Emotional Speech Set)
- SAVEE (Surrey Audio-Visual Expressed Emotion)
- Each dataset contains speech samples labeled by emotion, which are used to train the model. The primary emotions usually detected include happy, sad, angry, neutral, and fearful.

## Key Components

1. Data Preprocessing
Audio Feature Extraction: Extract features from the raw audio, such as Mel-Frequency Cepstral Coefficients (MFCCs), Mel spectrogram, or Chroma features, which effectively capture the nuances in tone and pitch.
Padding and Normalization: Pad audio sequences to a fixed length for consistency and normalize the features to aid model performance.

3. Model Architecture
The model leverages the sequential processing capability of LSTM layers to analyze temporal dependencies in audio data:
- LSTM Layers: Capture patterns in the sequential features derived from the audio signals.
- Dense Layers: Fully connected layers following LSTM to further refine the learned features.
- Output Layer: Uses a softmax activation for multi-class classification, identifying the predicted emotion.
  
3. Model Training
Loss Function: Categorical Cross-Entropy is used for multi-class classification tasks.
Optimizer: Adam optimizer, often with learning rate scheduling to enhance convergence.
Metrics: Accuracy is the primary metric, and additional metrics such as precision, recall, and F1-score help assess model performance across different emotion classes.

5. Model Evaluation
Confusion Matrix: Displays the accuracy of each emotion class prediction, showing where the model confuses different emotions.
Precision, Recall, F1-Score: Evaluates the model’s accuracy in predicting each emotion, providing insights into specific areas for improvement.
ROC-AUC: If binary emotions are targeted (e.g., positive vs. negative), the ROC-AUC metric is also evaluated.

7. Deployment
An inference pipeline is created to process new audio samples, apply the trained model, and output the predicted emotion. The model can be deployed in applications like real-time virtual assistants, emotion-based music recommendation systems, or mental health tools.

## Project Requirements

Libraries: TensorFlow/Keras for building and training the model, Librosa for audio processing, and Scikit-learn for metrics and evaluation.
Hardware: A GPU is recommended for faster processing, especially when training on large datasets.

## Conclusion

The Speech Emotion Recognition model using LSTM demonstrates how neural networks can effectively capture emotional tones in speech. This model serves as a foundational step toward developing applications that respond to users’ emotions, providing a personalized and empathetic user experience.
