# 🏠 House Price Predictor — Streamlit App

A beautiful, interactive house price prediction web app built with Streamlit.

## 📁 Project Files

```
streamlit_app/
│
├── app.py                 ← Main Streamlit application
├── Housing.csv            ← Dataset
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
│
└── models/
    ├── house_price_model.pkl  ← Trained Random Forest model
    ├── scaler.pkl             ← StandardScaler
    └── model_info.json        ← Model metadata & metrics
```

---

## 🚀 How to Run Locally

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```

### Step 3 — Open browser
```
http://localhost:8501
```

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your GitHub repo and set main file to `app.py`
5. Click **Deploy** — done! 🎉

---

## 🖥️ App Features

| Tab | What's inside |
|-----|--------------|
| 🔮 Predict Price | Interactive sliders & dropdowns, instant prediction, price range, percentile comparison |
| 📊 Data Explorer | Raw data table, charts, correlation heatmap |
| 📈 Model Insights | Feature importance, model metrics explained |

---

## 🤖 Model Info

- **Algorithm:** Random Forest Regressor (GridSearch Tuned)
- **R² Score:** 0.61
- **MAE:** ~₹10.3 Lakh
- **Training Data:** 436 houses (80%)
- **Test Data:** 109 houses (20%)
