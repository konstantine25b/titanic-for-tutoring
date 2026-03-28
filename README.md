# 🚢 Titanic - Machine Learning from Disaster

## 📌 კონკურსის მიმოხილვა

Kaggle-ის Titanic კონკურსის მიზანია მგზავრების გადარჩენის (Survived: 0/1) პროგნოზირება გემის კატასტროფის დროს. ეს ბინარული კლასიფიკაციის ამოცანაა, სადაც შეფასების ძირითადი მეტრიკაა **Accuracy**.

---

## 🧠 ჩემი მიდგომა

მონაცემები წინასწარ დამუშავებული იყო (Age/Fare ნორმალიზებული, Pclass/Title/Embarked OHE-კოდირებული). ამიტომ Feature Engineering-ში ჯერ განვაახლე კატეგორიული სვეტები OHE სვეტებიდან, შემდეგ გავაკეთე ახალი კოდირება — OHE და WOE — leakage-ის გარეშე (train split-ზე fit, val/test-ზე transform). Feature Selection-ისთვის გამოვიყენე RFE. ვატარებდი ექსპერიმენტებს Logistic Regression და Random Forest მოდელებით, ყოველი run ავტომატურად ილოგება MLflow-ში DagsHub-ზე.

---

## 📁 რეპოზიტორიის სტრუქტურა

```
titanic-for-tutoring/
│
├── data/
│   ├── train_data.csv        ← სასწავლო მონაცემები
│   └── test_data.csv         ← სატესტო მონაცემები
│
├── model_experiment.ipynb    ← ძირითადი ექსპერიმენტის notebook
├── model_inference.ipynb     ← პროგნოზირების notebook
├── submission.csv            ← Kaggle-ზე ასატვირთი შედეგები
└── README.md
```

---

## 📄 ფაილების აღწერა

| ფაილი                    | აღწერა                                                                          |
| ------------------------ | ------------------------------------------------------------------------------- |
| `model_experiment.ipynb` | Cleaning → Feature Engineering → Feature Selection → Training + MLflow logging  |
| `model_inference.ipynb`  | MLflow Registry-დან საუკეთესო მოდელის ჩამოტვირთვა, preprocess(), submission.csv |
| `data/train_data.csv`    | 793 მგზავრის მონაცემი (წინასწარ დამუშავებული)                                   |
| `data/test_data.csv`     | 101 მგზავრის სატესტო მონაცემი                                                   |

---

## 🧼 Data Cleaning

მონაცემები წინასწარ გაწმენდილი იყო, ამიტომ Cleaning ეტაპი მოიცავდა:

- `PassengerId` სვეტის წაშლა (ID, არ ატარებს ინფორმაციას)
- დუბლიკატი სტრიქონების წაშლა (`drop_duplicates`)
- დარჩენილი NaN მნიშვნელობების შევსება სვეტის **median**-ით

```python
def clean(df):
    df.drop(columns=['PassengerId'], errors='ignore', inplace=True)
    df.drop_duplicates(inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df
```

---

## 🧬 Feature Engineering

### ✦ Train/Val Split — პირველ ყოვლისა

Split ხდება **encoding-ამდე**, რათა თავიდან ავიცილოთ data leakage: ყველა encoder fit-დება მხოლოდ train split-ზე, val და test-ზე კი მხოლოდ transform.

```
Train split: 80%  |  Val split: 20%  (stratified by Survived)
```

### ✦ კატეგორიული სვეტების აღდგენა

მონაცემებში Pclass, Title, Embarked უკვე OHE-ადაა კოდირებული (Pclass_1/2/3, Title_1-4, Emb_1-3). ვაღდგენთ პირვანდელ კატეგორიულ მნიშვნელობებს `idxmax`-ით:

```python
df['Pclass']   = df[['Pclass_1','Pclass_2','Pclass_3']].idxmax(axis=1).str.replace('Pclass_','').astype(int)
df['Title']    = df[['Title_1','Title_2','Title_3','Title_4']].idxmax(axis=1).str.replace('Title_','').astype(int)
df['Embarked'] = df[['Emb_1','Emb_2','Emb_3']].idxmax(axis=1).str.replace('Emb_','').astype(int)
```

### ✦ ახალი Feature: IsAlone

```python
df['IsAlone'] = (df['Family_size'] == 0).astype(int)
```

მარტოხელა მგზავრობა სტატისტიკურად დაკავშირებულია გადარჩევასთან.

### ✦ One-Hot Encoding (OHE)

გამოყენება: `Sex`, `Pclass`, `Title`, `Embarked`

- `pd.get_dummies` fit-დება **train**-ზე
- val და test სვეტები `reindex`-ით align-დება train-ის სვეტებთან (დაკარგული სვეტები → 0)

შედეგი: ~16 ბინარული feature

### ✦ Weight of Evidence (WOE) Encoding

გამოყენება: `Sex`, `Pclass`, `Title`, `Embarked`

- ბიბლიოთეკა: `category_encoders.WOEEncoder(regularization=1.0)`
- ფორმულა: `WOE = ln(P(X|Y=1) / P(X|Y=0))`
- WOE ბუნებრივია ბინარული კლასიფიკაციისთვის — პირდაპირ ასახავს კატეგორიის "სასარგებლობას" გადარჩევის პროგნოზისთვის
- fit მხოლოდ train-ზე, transform val და test-ზე

OHE-სგან განსხვავებით WOE ერთი სვეტით ანაცვლებს კატეგორიულ სვეტს (კომპაქტური).

---

## 🔍 Feature Selection

### Recursive Feature Elimination (RFE)

- **Estimator:** RandomForestClassifier (100 ხე)
- **OHE-სთვის:** 15 საუკეთესო feature
- **WOE-სთვის:** 8 საუკეთესო feature
- fit ხდება **train split-ზე** მხოლოდ

RFE iteratively ამოიღებს ყველაზე სუსტ feature-ებს feature importance-ის მიხედვით. Ranking plot ილოგება ყოველ გაშვებაზე.

---

## 🧪 Training

ყოველი run-ი: **5-fold Stratified CV** train split-ზე + val metrics held-out val-ზე.

### 🔹 Logistic Regression — OHE vs WOE

| Run | Encoding | C | cv_auc | val_auc | val_acc | val_f1 |
|---|---|---|---|---|---|---|
| LR_OHE_C0.01 | OHE | 0.01 | 0.8495 | 0.8331 | 0.7635 | 0.6957 |
| LR_OHE_C0.1  | OHE | 0.1  | 0.8581 | 0.8426 | 0.8041 | 0.7521 |
| **LR_OHE_C1.0** | **OHE** | **1.0** | **0.8575** | **0.8452** | **0.8108** | **0.7627** |
| LR_OHE_C10.0 | OHE | 10.0 | 0.8574 | 0.8445 | 0.8041 | 0.7563 |
| LR_WOE_C0.01 | WOE | 0.01 | 0.8383 | 0.8215 | 0.7297 | 0.6491 |
| LR_WOE_C0.1  | WOE | 0.1  | 0.8464 | 0.8354 | 0.7635 | 0.6957 |
| LR_WOE_C1.0  | WOE | 1.0  | 0.8441 | 0.8348 | 0.7703 | 0.7018 |
| LR_WOE_C10.0 | WOE | 10.0 | 0.8443 | 0.8356 | 0.7703 | 0.7018 |

Logistic Regression პირველ მოდელად გამოვიყენე, რადგან კარგი baseline-ია. C პარამეტრი აკონტროლებს regularization-ს — პატარა C = ძლიერი regularization. OHE კოდირება სტაბილურად უკეთეს შედეგს იძლეოდა WOE-სთან შედარებით.

### 🔹 Random Forest — OHE vs WOE

| Run | Encoding | n_estimators | max_depth | cv_auc | val_auc | val_acc | val_f1 |
|---|---|---|---|---|---|---|---|
| RF_OHE_n100_d6 | OHE | 100 | 6 | 0.8529 | 0.8437 | **0.8176** | 0.7477 |
| RF_WOE_n100_d6 | WOE | 100 | 6 | 0.8549 | 0.8431 | 0.7905 | 0.7103 |

Random Forest val_auc-ით (0.8437) ოდნავ ჩამოუვარდება LR-ს, მაგრამ val_acc-ზე (0.8176) საუკეთესო შედეგი ჰქონდა ყველა run-ს შორის. `max_depth=6` ზღუდავს overfitting-ს.

---

## 📊 MLflow Tracking

### ბმული

DagsHub MLflow: [https://dagshub.com/konstantine25b/titanic-for-tutoring/experiments](https://dagshub.com/konstantine25b/titanic-for-tutoring/experiments)

### ჩაწერილი მეტრიკები

| მეტრიკა                      | აღწერა                                                               |
| ---------------------------- | -------------------------------------------------------------------- |
| `cv_acc_mean` / `cv_acc_std` | 5-fold CV Accuracy — საშუალო და სტანდარტული გადახრა                  |
| `cv_auc_mean` / `cv_auc_std` | 5-fold CV ROC-AUC — მთავარი CV მეტრიკა                               |
| `cv_f1_mean` / `cv_f1_std`   | 5-fold CV F1-score                                                   |
| `val_accuracy`               | Held-out val set accuracy                                            |
| `val_f1`                     | Held-out val set F1                                                  |
| `val_auc`                    | Held-out val set ROC-AUC — **საუკეთესო მოდელის შერჩევის კრიტერიუმი** |

### ჩაწერილი პარამეტრები

`model`, `encoding`, `n_features`, `val_size` + მოდელ-სპეციფიური (`C`, `n_estimators`, `max_depth`)

### Artifacts თითო Run-ზე

- Confusion Matrix + ROC Curve plot (val set-ზე)
- Feature Importance plot (RF-ისთვის)
- Serialized sklearn model (`model/`)

### ⭐️ საუკეთესო მოდელი

საუკეთესო მოდელი შეირჩევა `val_auc`-ის მიხედვით:

| პარამეტრი | მნიშვნელობა |
|---|---|
| მოდელი | Logistic Regression (Pipeline + StandardScaler) |
| Encoding | OHE |
| C | 1.0 |
| Features | 15 (RFE-ით შერჩეული) |
| cv_auc | **0.8575** |
| val_auc | **0.8452** |
| val_accuracy | **0.8108** |
| val_f1 | **0.7627** |

ხელახლა ტრენინგდება **train + val**-ის მთლიანობაზე და რეგისტრირდება MLflow Model Registry-ში სახელით `titanic-best-model`.

---

## 🔮 Inference

`model_inference.ipynb` ასრულებს შემდეგს:

1. MLflow Model Registry-დან `titanic-best-model/latest` ჩამოტვირთვა
2. Preprocessing artifacts (WOE encoder, selected features) DagsHub-იდან ჩამოტვირთვა — **local cache**-ით სისწრაფისთვის
3. `preprocess(df)` ფუნქცია — raw test data → model-ready numpy array
4. `submission.csv` გენერაცია Kaggle-ზე ასატვირთად

---

## 🛠 გაშვების ინსტრუქცია

1. `data/` საქაღალდეში მოათავსეთ `train_data.csv` და `test_data.csv`
2. DagsHub credentials შეავსეთ ორივე notebook-ში
3. გაუშვით `model_experiment.ipynb` — ლოგავს ყველა run-ს DagsHub-ზე
4. გაუშვით `model_inference.ipynb` — ქმნის `submission.csv`
