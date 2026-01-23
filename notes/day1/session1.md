# Naive Bayers Probabilistic classifier

- A classifier is a machine learning model that assigns a label (category) to input data. In simple words a classifier decides which class something belongs to.
- Naive Bayes is a machine learning classification algorithm that predicts the category of a data point using probability.
- It assumes that all features are independent of each other. Naive Bayes performs well in many real-world applications such as spam filtering, document categorisation and sentiment analysis.

## Bayes' theorem

<img src="./resources/Bayes'%20theorem.png" width="500" height="400">

## Why it is called Naive?

Because of the independence assumption 👇

Naive assumption:
All features are independent of each other given the class

```
Example (Email Spam):

Suppose an email contains words:

“win”

“money”

“offer”

Naive Bayes assumes:

P(win, money, offer | Spam)
= P(win | Spam) × P(money | Spam) × P(offer | Spam)

```

👉 In reality, these words are not independent,
but surprisingly Naive Bayes still works very well.

## How Classification Works (Step-by-Step)

Let’s say we have two classes:

- **Spam**
- **Not Spam**

For a new email:

### Step 1: Compute probability for each class

**Score(Spam)**

$$
\text{Score(Spam)} = P(\text{Spam}) \times P(\text{words} \mid \text{Spam})
$$

**Score(Not Spam)**

$$
\text{Score(Not Spam)} = P(\text{Not Spam}) \times P(\text{words} \mid \text{Not Spam})
$$

### Step 2: Compare scores

Whichever score is higher → **predicted class**

---

## Simple Numerical Example

### Dataset

Suppose:

- **60%** emails are **Spam**
- **40%** emails are **Not Spam**

### Word probabilities

| Word  | P(word \| Spam) | P(word \| Not Spam) |
| ----- | --------------- | ------------------- |
| win   | 0.4             | 0.05                |
| money | 0.3             | 0.02                |

### New email

**win money**

**Spam score**

$$
0.6 \times 0.4 \times 0.3 = 0.072
$$

**Not Spam score**

$$
0.4 \times 0.05 \times 0.02 = 0.0004
$$

✅ **Spam score is higher → Classified as Spam**

---

## Why Naive Bayes Works Well (Despite Being Naive)

✔ Very fast  
✔ Works well with high-dimensional data  
✔ Requires small training data  
✔ Robust to irrelevant features

That’s why it’s extremely popular in **text-based problems**.

---

## Common Applications

### 📨 Text Classification

- Spam vs Not Spam
- Topic classification (sports, politics, tech)

### 😊 Sentiment Analysis

- Positive / Negative / Neutral reviews
- Twitter sentiment analysis

### 🔍 Document Categorization

- News articles
- Support ticket routing

### ⚠️ Medical Diagnosis (basic level)

- Disease probability based on symptoms

---

## Types of Naive Bayes (Preview)

We choose the variant based on data type:

| Variant        | Used For                  |
| -------------- | ------------------------- |
| Gaussian NB    | Continuous numerical data |
| Multinomial NB | Text data (word counts)   |
| Bernoulli NB   | Binary features (yes/no)  |

👉 We’ll cover **Gaussian NB in Session 2** and **Multinomial NB in Session 3**.

---

## Advantages & Limitations

### ✅ Advantages

- Extremely fast
- Easy to implement
- Scales well
- Works great for NLP

### ❌ Limitations

- Independence assumption is unrealistic
- Poor performance if features are highly correlated
- Zero probability issue (handled using **Laplace smoothing**)

---

## Key Takeaways (Session 1)

- Naive Bayes is a **probability-based classifier**
- Based on **Bayes’ Theorem**
- Assumes **feature independence**
- Very effective for **text classification**
- Foundation for **Gaussian & Multinomial Naive Bayes**
