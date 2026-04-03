import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path
from itertools import islice
import random
import sklearn




NEW_DATA_PATH = Path("Unit1/Week3/sms+spam+collection/SMSSpamCollection")
with open(NEW_DATA_PATH, 'r') as file:
    for line in islice(file, 99, 110):  # 99 because it's 0-indexed
        label, message = line.strip().split(maxsplit=1)
        #print(f"Label: {label}, Message: {message}")

NEW_DATA: List[Tuple[str, int]] = []

with open(NEW_DATA_PATH, 'r') as file:
    for line in file:
        label, message = line.strip().split(maxsplit=1)
        label = 1 if label == 'spam' else 0
        NEW_DATA.append((message,label))

#print("NEW_DATA:",NEW_DATA[0:4])
# Shuffle the data
random.seed(42)
random.shuffle(NEW_DATA)

# Split 80/20
split_point = int(len(NEW_DATA) * 0.8)
train_data = NEW_DATA[:split_point]
test_data = NEW_DATA[split_point:]
# ============================================================
# 1) DATASET: 30 short texts (made up)
# label: 1 = spam, 0 = ham
# ============================================================



DATA: List[Tuple[str, int]] = [
    ("CONGRATS! You won a $500 gift card. Click to claim now!", 1),
    ("Urgent: Your account is suspended. Verify here: http://bit.ly/verify", 1),
    ("Limited time offer!!! Buy 1 get 2 free. Shop today!", 1),
    ("You have been selected for a FREE vacation. Reply YES to win.", 1),
    ("Earn money fast from home. No experience needed!", 1),
    ("Final notice: pay your invoice immediately to avoid fees.", 1),
    ("WIN a brand new phone—just enter your details at our link.", 1),
    ("Get cheap meds without prescription. Order now!", 1),
    ("Exclusive deal: 0% APR approved instantly. Apply now.", 1),
    ("You are pre-approved for a cash loan. Same-day deposit!", 1),
    ("Act now! Your prize expires in 2 hours. Click here.", 1),
    ("Hot singles in your area waiting—sign up free!", 1),
    ("Claim your crypto bonus today. Limited slots!", 1),
    ("Lowest price guaranteed!!! Visit our website for discounts.", 1),
    ("Congratulations winner: confirm your shipping address to receive reward.", 1),

    ("Hey, are we still meeting for coffee at 3?", 0),
    ("Can you review my draft before tomorrow’s class?", 0),
    ("I’ll be late—traffic is worse than usual.", 0),
    ("Don’t forget the staff meeting on Tuesday at 10am.", 0),
    ("What time does the movie start tonight?", 0),
    ("Thanks for helping with the homework problem!", 0),
    ("I uploaded the slides to the course folder.", 0),
    ("Can you pick up milk and eggs on your way home?", 0),
    ("The exam covers chapters 1 through 4 and some trig review.", 0),
    ("Great job on the presentation—students liked the examples.", 0),
    ("Lunch was fun—let’s do it again next week.", 0),
    ("Your package arrived; I left it by the front door.", 0),
    ("Please call me when you get a chance.", 0),
    ("The internet is down in my building again.", 0),
    ("Reminder: dentist appointment Friday at 2:30.", 0),
]
#print("DATA:", DATA[0:1])
def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-z0-9']+", text)

@dataclass
class NaiveBayesModel:
    log_prior_spam: float
    log_prior_ham: float
    log_likelihood_spam: Dict[str, float]
    log_likelihood_ham: Dict[str, float]
    log_unknown_spam: float
    log_unknown_ham: float

def train_naive_bayes(data: List[Tuple[str, int]], alpha: float = 1.0) -> NaiveBayesModel:
    spam_texts = [t for t, y in data if y == 1]
    ham_texts = [t for t, y in data if y == 0]

    log_prior_spam = math.log(len(spam_texts) / len(data))
    log_prior_ham = math.log(len(ham_texts) / len(data))

    spam_counts = Counter()
    ham_counts = Counter()
    #print("DEBUG",spam_texts[0:3])
    for text in spam_texts:
        spam_counts.update(tokenize(text))
    for text in ham_texts:
        ham_counts.update(tokenize(text))

    vocab = set(spam_counts) | set(ham_counts)
    V = len(vocab)

    spam_total = sum(spam_counts.values())
    ham_total = sum(ham_counts.values())

    denom_spam = spam_total + alpha * V
    denom_ham = ham_total + alpha * V

    log_likelihood_spam = {}
    log_likelihood_ham = {}

    for w in vocab:
        log_likelihood_spam[w] = math.log((spam_counts[w] + alpha) / denom_spam)
        log_likelihood_ham[w] = math.log((ham_counts[w] + alpha) / denom_ham)

    log_unknown_spam = math.log(alpha / denom_spam)
    log_unknown_ham = math.log(alpha / denom_ham)

    return NaiveBayesModel(
        log_prior_spam,
        log_prior_ham,
        log_likelihood_spam,
        log_likelihood_ham,
        log_unknown_spam,
        log_unknown_ham,
    )

def predict(model: NaiveBayesModel, text: str) -> int:
    tokens = tokenize(text)
    log_spam = model.log_prior_spam
    log_ham = model.log_prior_ham

    for w in tokens:
        log_spam += model.log_likelihood_spam.get(w, model.log_unknown_spam)
        log_ham += model.log_likelihood_ham.get(w, model.log_unknown_ham)

    return 1 if log_spam > log_ham else 0

if __name__ == "__main__":
    model = train_naive_bayes(train_data)
    samples = [
        "Reminder: our project meeting is at 9 tomorrow.",
        "You won a prize—click to claim your reward now!",
    ]
    y_message = [message for message, label in test_data]
    y_true = [label for message, label in test_data]
    y_preds = []
    for s, label in test_data[10:20]:
        y_pred = predict(model, s)
        print(s, "->", "SPAM" if y_pred else "HAM", "Actual:", "ham" if label == 0 else "spam")
    for s, label in test_data:
        y_predd = predict(model, s)
        y_preds.append(y_predd)
    print(sklearn.metrics.classification_report(y_true, y_preds))
    cm = sklearn.metrics.confusion_matrix(y_true, y_preds)
    print("Confusion Matrix:")
    print(f"              Predicted Ham  Predicted Spam")
    print(f"Actual Ham    {cm[0,0]:<14} {cm[0,1]}")
    print(f"Actual Spam   {cm[1,0]:<14} {cm[1,1]}")