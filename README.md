# Titanic — Machine Learning from Disaster

## კონკურსის მიმოხილვა

Kaggle-ის კლასიკური კლასიფიკაციის კონკურსი — Titanic-ის გადარჩენილების პროგნოზი.  
მიზანი: ბინარული კლასიფიკატორი, რომელიც ამბობს გადარჩა თუ არა მგზავრი (0/1).  
მონაცემები: `train.csv` (891 ჩანაწერი), `test.csv` (418 ჩანაწერი).  
შეფასების მეტრიკა: **Accuracy**.

---

## ჩემი მიდგომა

1. მონაცემების გასუფთავება და missing value-ების დამუშავება
2. Feature Engineering — ახალი ნიშნების გამოტანა
3. ორი სხვადასხვა კოდირების სქემა: **One-Hot Encoding** და **WOE Encoding**
4. Feature Selection **RFE**-ით
5. სხვადასხვა მოდელებისა და ჰიპერპარამეტრების ტესტირება
6. ყველა გაშვების ლოგირება **MLflow** / **DagsHub**-ზე
7. საუკეთესო მოდელის MLflow Model Registry-ში რეგისტრაცია
8. `model_inference.ipynb`-ში Registry-დან ჩამოტვირთვა და `submission.csv`-ის გენერაცია

---

## რეპოზიტორიის სტრუქტურა

```
├── model_experiment.ipynb       # მთავარი ექსპერიმენტის notebook
├── model_inference.ipynb        # inference notebook (submission.csv)
├── preprocessing_artifacts.pkl  # WOE encoder + selected features (runtime artifact)
├── README.md                    # ეს ფაილი
└── mlruns/                      # MLflow local tracking (local გაშვებისას)
```

---

## ფაილების განმარტება

| ფაილი | აღწერა |
|---|---|
| `model_experiment.ipynb` | მთელი pipeline: Cleaning → Feature Engineering → Feature Selection → Training + MLflow logging |
| `model_inference.ipynb` | Registry-დან მოდელის ჩამოტვირთვა, test set-ზე predict, submission.csv |
| `preprocessing_artifacts.pkl` | pickle-ირებული WOE encoder, შერჩეული ფიჩერების სიები |
| `README.md` | პროექტის დოკუმენტაცია |

---

## Data Cleaning

**Missing Value-ების დამუშავება:**

| სვეტი | სტრატეგია |
|---|---|
| `Age` | Median-ით შევსება `Pclass` × `Sex` ჯგუფების მიხედვით |
| `Fare` | Median-ით შევსება `Pclass`-ის მიხედვით |
| `Embarked` | Mode-ით შევსება (2 missing) |
| `Cabin` | პირველი სიმბოლო (`Deck`) ამოღება; დანარჩენი `Unknown`; სვეტი წაშლა |

**სხვა ოპერაციები:**
- `Name`, `Ticket`, `PassengerId` სვეტები წაშლა (ID-ები და ტექსტი)
- `Title` ამოღება `Name`-დან regex-ით; იშვიათი ტიტულები გაერთიანება `'Rare'` კლასში

---

## Feature Engineering

**ახალი ფიჩერები:**

| ფიჩერი | ლოგიკა |
|---|---|
| `Title` | სახელიდან ამოღებული ტიტული (Mr, Mrs, Miss, Master, Rare) |
| `FamilySize` | `SibSp + Parch + 1` |
| `IsAlone` | `FamilySize == 1` → 1, სხვა → 0 |
| `AgeBin` | ასაკობრივი ბინები: Child/Teen/Adult/MidAge/Senior |
| `FareBin` | კვარტილებით: Low/Mid/High/VHigh |
| `Deck` | Cabin-ის პირველი ასო; Unknown თუ კაბინა არ არის |

### კატეგორიული ცვლადების კოდირება

#### One-Hot Encoding (OHE)
გამოყენება: `Sex`, `Embarked`, `Title`, `Deck`, `AgeBin`, `FareBin`  
მიდგომა: `pd.get_dummies` train+test-ის კომბინირებით (column alignment)  
შედეგი: ~35 ბინარული სვეტი

#### Weight of Evidence (WOE) Encoding
გამოყენება: `Sex`, `Embarked`, `Title`, `Deck`, `AgeBin`, `FareBin`, `Pclass`  
ბიბლიოთეკა: `category_encoders.WOEEncoder`  
ფორმულა: `WOE = ln(P(X|Y=1) / P(X|Y=0))`  
WOE ბუნებრივია ბინარული კლასიფიკაციისთვის — პირდაპირ ასახავს კატეგორიის "სასარგებლობას" სამიზნე ცვლადის პროგნოზისთვის

---

## Feature Selection

### Recursive Feature Elimination (RFE)

- **Estimator:** RandomForestClassifier (100 ხე)
- **OHE-სთვის:** 15 საუკეთესო ფიჩერი
- **WOE-სთვის:** 10 საუკეთესო ფიჩერი
- **Step:** 1 (ერთი ფიჩერი ამოიღება ყოველ ეტაპზე)

RFE feature importance-ზე დაყრდნობით iteratively ამოიღებს ყველაზე სუსტ ფიჩერებს. ვიყენებთ `rfe.ranking_`-ს ვიზუალიზაციისთვის.

**შედეგი:** OHE-ით ამოვარჩიეთ 15/35, WOE-ით 10/17 ფიჩერი.

---

## Training

### ტესტირებული მოდელები

| მოდელი | კოდირება | ტესტირებული ჰიპერპარამეტრები |
|---|---|---|
| Logistic Regression | OHE + WOE | C ∈ {0.01, 0.1, 1.0, 10.0} |
| Random Forest | OHE + WOE | n_estimators ∈ {50,100,200}, max_depth ∈ {4,6,None} |
| XGBoost | OHE + WOE | learning_rate ∈ {0.05,0.1,0.2}, max_depth ∈ {3,5,7} |
| LightGBM | OHE + WOE | num_leaves ∈ {15, 31, 63} |

### Hyperparameter ოპტიმიზაციის მიდგომა

გამოვიყენეთ manual grid search — ყველა კომბინაცია გავუშვით, ყოველი run-ი დავალოგეთ MLflow-ზე.  
შეფასება: **5-fold Stratified Cross-Validation** სამი მეტრიკით (Accuracy, ROC-AUC, F1).

### საბოლოო მოდელის შერჩევის დასაბუთება

**კრიტერიუმი:** `cv_auc` (cross-validation ROC-AUC) — ყველაზე robust მეტრიკა დაუბალანსებელი კლასებისთვის.  
`results_df.sort_values('cv_auc').iloc[0]`-ის გამარჯვებული run-ი ავტომატურად რეგისტრირდება Model Registry-ში `titanic-best-model` სახელით.

---

## MLflow Tracking

### ბმული
DagsHub-ის MLflow UI: `https://dagshub.com/{USERNAME}/{REPO}/experiments`  
(Local-ისთვის გაუშვი `mlflow ui` პროექტის papkიდან და გახსენი `http://localhost:5000`)

### ჩაწერილი მეტრიკები

| მეტრიკა | აღწერა |
|---|---|
| `cv_acc_mean` / `cv_acc_std` | 5-fold CV Accuracy საშუალო/სტდ |
| `cv_auc_mean` / `cv_auc_std` | 5-fold CV ROC-AUC საშუალო/სტდ |
| `cv_f1_mean` / `cv_f1_std` | 5-fold CV F1 საშუალო/სტდ |
| `val_accuracy` | Validation set (20%) accuracy |
| `val_f1` | Validation set F1 |
| `val_auc` | Validation set ROC-AUC |

### ჩაწერილი პარამეტრები

`model`, `encoding`, `n_features` + მოდელ-სპეციფიური პარამეტრები (C, n_estimators, max_depth, learning_rate, num_leaves)

### Artifacts თითო Run-ზე

- `{run_name}_plots.png` — Confusion Matrix + ROC Curve
- `{run_name}_importance.png` — Feature Importance (tree-based მოდელებისთვის)
- `model/` — serialized sklearn model

### საუკეთესო მოდელის შედეგები

საუკეთესო მოდელი ავტომატურად ირეგისტრირება MLflow Model Registry-ში (`titanic-best-model`, version `latest`) და `model_inference.ipynb` მას იქიდანვე ტვირთავს.

---

## გაშვების ინსტრუქცია (Kaggle)

1. Notebook-ების ატვირთვა Kaggle-ზე
2. Titanic dataset-ის დამატება (`/kaggle/input/titanic/`)
3. `model_experiment.ipynb`-ის გაშვება — ლოგავს ყველა ექსპერიმენტს, ინახავს `preprocessing_artifacts.pkl`
4. `model_inference.ipynb`-ის გაშვება — ტვირთავს საუკეთესო მოდელს, ქმნის `submission.csv`
5. `submission.csv`-ის Kaggle-ზე ატვირთვა

**DagsHub remote tracking-ისთვის:** `USE_DAGSHUB = True` და შეავსე credentials ორივე notebook-ში.
