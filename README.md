# 📝 Text Intelligence: Sentiment Analysis & News Classification

## 📌 Project Overview  
This project focuses on **Natural Language Processing (NLP)** tasks, covering two key areas:

### 🔹 Part 1: Sentiment Analysis (IMDb Movie Reviews)  
- Predicts whether a movie review is **positive** or **negative** using machine learning models.
- **Models Used**: Logistic Regression, LSTM.
- **Performance**: Logistic Regression achieves **88% accuracy**.
- **Applications**: Helps movie producers, critics, and platforms like IMDb analyze audience sentiment.

### 🔹 Part 2: News Classification  
- Classifies news articles into predefined categories such as **sports, politics, and technology**.
- **Models Used**: Random Forest, Word Cloud, Stemming.
- **Applications**: Enables news organizations and social media platforms to **automate content categorization**.

---

## 📂 Dataset  
### **IMDb Sentiment Analysis Dataset**  
- **Text Review**: Contains raw movie reviews.
- **Sentiment Label**: Either "positive" or "negative".

### **News Classification Dataset**  
- **Article Text**: News articles related to different domains.
- **Category Label**: Classified into "Sports", "Politics", "Technology".

---

## 🏆 Sentiment Analysis (IMDb Reviews)  
### 🔹 Logistic Regression Model (88% Accuracy)  
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Train model
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)

# Predictions
y_pred = log_reg.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### 🔹 LSTM Model Training  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Define LSTM Model
model = Sequential([
    Input(shape=(1, X_train_tfidf.shape[2])),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
```

---

## 📰 News Article Classification  
### 🔹 Word Cloud for Frequent Words  
```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Generate Word Cloud
all_words = " ".join([sentence for sentence in df1['clean_text']])
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(all_words)

# Plot
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

### 🔹 Text Preprocessing with Stemming  
```python
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
df1['tokens'] = df1['tokens'].apply(lambda sentence: [stemmer.stem(word) for word in sentence])
df1.head()
```

### 🔹 Random Forest Classification  
```python
from sklearn.ensemble import RandomForestClassifier

# Train model
forest = RandomForestClassifier()
model = forest.fit(X_train_vector, y_train)

# Predictions
y_pred = model.predict(X_test_vector)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## 📊 Results & Insights  
🔹 **Logistic Regression achieves 88% accuracy in Sentiment Analysis**.
🔹 **LSTM model enhances performance but requires more computational power**.
🔹 **Random Forest is effective for classifying news articles into categories**.
🔹 **Word Cloud visualization helps identify common words in different news categories**.

---

## 📂 Resources  
📌 **Sentiment Analysis**: [https://drive.google.com/file/d/1FZjLvaGwDSJZTLzdMbz_3UGIzZVyXtG8/view?usp=sharing]  
📌 **News Classification**: [https://drive.google.com/file/d/1Af_jacu4zM20e3KSEQc2g0QBgvDD_DGt/view?usp=sharing]  

---

🚀 **Future Improvements**: Explore transformer models like **BERT** for better sentiment classification, and try **TF-IDF + Neural Networks** for improved news classification.

