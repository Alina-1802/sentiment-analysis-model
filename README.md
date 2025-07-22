# Sentiment Classification Model with LSTM

A deep learning model for emotion classification in text data, developed as part of a university course on **Artificial Intelligence in Business Analytics**.

<img width="400" height="350" alt="confusion_matrix4" src="https://github.com/user-attachments/assets/7d39c3a3-a470-48e1-979d-db8b6206ecd8" />

## 📋 Description

This project builds a sentiment classification model using an **LSTM neural network** to recognize emotions based on raw text. The solution supports businesses in understanding user sentiment and responding more effectively.

The model was trained on a labeled dataset of emotional expressions and optimized through hyperparameter tuning.

## 💼 Business Context

Emotion classification in text has various business applications, such as:

- Customer opinion mining (e.g., product or service reviews)
- Monitoring public mood on social media
- Detecting potential PR crises
- Enhancing chatbot responses through emotion-aware dialog

By identifying user emotions accurately, companies can make better decisions, improve customer experience, and build stronger user relationships.

## 📊 Dataset

- **Name**: *Emotions Dataset*
- **Source**: [Kaggle](https://www.kaggle.com/datasets/datasets/emotions)
- **Size**: ~400,000 records  
- **Columns**: `id`, `text`, `label` (emotion)
- **Classes**:  
  - `0` - sadness  
  - `1` - joy  
  - `2` - love  
  - `3` - anger  
  - `4` - fear  
  - `5` - surprise  
- No missing values  
- Labels are imbalanced (sadness and joy are dominant)

## 🧹 Data Preparation

- Text cleaning & tokenization  
- Label encoding  
- Exploratory analysis (e.g., class distribution, text length stats)  
- Median input length: **17 words**

## 🧠 Model Architecture

The core model is a **Long Short-Term Memory (LSTM)** network:

- Embedding layer  
- LSTM layer  
- Dropout for regularization  
- Dense output layer with softmax activation  

## 🧪 Modeling Strategy

Five model variants were trained and compared:

- **Model 1**: baseline LSTM  
- **Model 2-3**: tuned dropout and epochs  
- **Model 4-5**: increased input sequence length to 40 tokens

## ✅ Key Results

- Final accuracy: **~93%**  
- Best performing: **Model 4** (longer input context)  
- Main improvements came from increasing input length, not from dropout or epoch count  
- Significant boost in F1-score for minority emotion classes  
- Confusion matrix showed drastic error reduction in ambiguous or underrepresented emotions

> 💡 Longer input sequences helped the model better understand the semantic context of each text.

## 🧰 Tools & Technologies

- Python  
- TensorFlow / Keras  
- NumPy, pandas, seaborn, matplotlib, scikit-learn

## 📈 Sample Visuals
<img width="300" height="250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9d9be7ed-356f-44ee-a6de-091933f8e79b" />
<img width="300" height="250" alt="confusion_matrix5" src="https://github.com/user-attachments/assets/2e0ab70d-4527-4266-a5a0-d8296a010c80" />
<img width="300" height="250" alt="confusion_matrix4" src="https://github.com/user-attachments/assets/83dffca4-8e7e-45ad-a409-f625bee9fcbd" />
<img width="300" height="250" alt="confusion_matrix3" src="https://github.com/user-attachments/assets/4bd3a693-0745-4666-ae3b-2e04e3809958" />
<img width="300" height="250" alt="confusion_matrix2" src="https://github.com/user-attachments/assets/390c4b86-7767-465b-b5dd-d6e453ceff28" /> 
<br><br>
<img width="400" height="350" alt="training_history" src="https://github.com/user-attachments/assets/c15b2556-b908-4839-a085-470e547d79ea" />
<img width="400" height="350" alt="training_history5" src="https://github.com/user-attachments/assets/0f2742dd-2afd-41ef-9b4c-ab3219e31e50" />
<img width="400" height="350" alt="training_history4" src="https://github.com/user-attachments/assets/3ad23dfb-b2cd-4cbe-bab9-39933cc353a4" />
<img width="400" height="350" alt="training_history3" src="https://github.com/user-attachments/assets/f61a0acc-8c62-487f-ba90-a9655ae1dda9" />
<img width="400" height="350" alt="training_history2" src="https://github.com/user-attachments/assets/9e82f5c7-e4c2-4aa5-9b70-69ba407a10ab" />

