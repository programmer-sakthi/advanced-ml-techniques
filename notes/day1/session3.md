# Multinomial Naive Bayes

- Multinomial Naive Bayes is one of the variation of Naive Bayes algorithm which is ideal for discrete data and is typically used in text classification problems.
- It models the frequency of words as counts and assumes each feature or word is multinomially distributed.
- MNB is widely used for tasks like classifying documents based on word frequencies like in spam email detection.

## How Does Multinomial Naive Bayes Work?

- In Multinomial Naive bayes the word "Naive" means that the method assumes all features like words in a sentence are independent from each other and "Multinomial" refers to how many times a word appears or how often a category occurs.
- It works by using word counts to classify text.
- The main idea is that it assumes each word in a message or feature is independent of each others.
- This means the presence of one word doesn't affect the presence of another word which makes the model easy to use.

# STEP-BY-STEP Multinomial Naive Bayes (Text Classification)

## 🎯 Problem

Classify an email as **Spam** or **Not Spam**

---

## STEP 1️⃣ Collect Training Data

| Email | Text                  | Class    |
| ----- | --------------------- | -------- |
| E1    | buy cheap now         | Spam     |
| E2    | limited offer buy now | Spam     |
| E3    | meeting tomorrow      | Not Spam |
| E4    | project meeting       | Not Spam |

👉 This is your **training dataset**

---

## STEP 2️⃣ Create Vocabulary (Bag of Words)

```
Vocabulary =
[buy, cheap, now, limited, offer, meeting, tomorrow, project]
```

Vocabulary size **V = 8**

---

## STEP 3️⃣ Convert Text → Numbers (BoW Matrix)

| Email | buy | cheap | now | limited | offer | meeting | tomorrow | project | Class    |
| ----- | --- | ----- | --- | ------- | ----- | ------- | -------- | ------- | -------- |
| E1    | 1   | 1     | 1   | 0       | 0     | 0       | 0        | 0       | Spam     |
| E2    | 1   | 0     | 1   | 1       | 1     | 0       | 0        | 0       | Spam     |
| E3    | 0   | 0     | 0   | 0       | 0     | 1       | 1        | 0       | Not Spam |
| E4    | 0   | 0     | 0   | 0       | 0     | 1       | 0        | 1       | Not Spam |

---

## STEP 4️⃣ Calculate Class Priors

P(Spam) = 2/4 = 0.5  
P(NotSpam) = 2/4 = 0.5

---

## STEP 5️⃣ Count Words Per Class

### 🔴 Spam Emails

Total words = **7**

### 🔵 Not Spam Emails

Total words = **4**

---

## STEP 6️⃣ Laplace Smoothing (α = 1)

P(word|class) = (count + 1) / (total_words + V)

---

## STEP 7️⃣ New Email

```
buy offer now
```

---

## STEP 8️⃣ Multinomial NB Formula

Score(C) = P(C) × ∏ P(word|C)

---

## STEP 9️⃣ Prediction

✅ **Spam**

---

## STEP 🔟 Why Logs

Avoid numerical underflow, same result.

---

## FINAL FLOW

Text → BoW → Priors → Word Probabilities → Score → Prediction
