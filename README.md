# Pockit — ML-Powered Personal Finance Tracker

A smart finance tracking web app that automatically extracts, categorizes, and analyzes your spending from bank statements using Machine Learning.


## Features

| Feature | Description |
|  Bank Statement Import | Upload PDF or Excel statements — transactions auto-extracted |
| Auto Categorization | Random Forest + keyword matching classifies Food, Transport, Shopping, Rent and more |
| Broke Day Predictor | Predicts which day of the month you'll run out of money |
| Impulse Buy Detector | Flags late-night and unplanned purchases |
| Spending Personality | Labels your spending style — Foodie, Impulsive, Balanced, Social Spender |
| Subscription Graveyard | Surfaces forgotten recurring charges |
| Manual Category Fix | Fix unknown UPI merchants with one click — system learns permanently |
| Interactive Dashboard | Donut chart, monthly trend line, category breakdown bars |
| Monthly Roast | Honest (slightly brutal) summary of your spending habits |
| Future You Warning | Projects your spending 6 and 12 months ahead |

---

## Tech Stack

- **Backend** — Python, Flask, Flask-CORS
- **ML** — Scikit-learn (Random Forest, Isolation Forest, Linear Regression)
- **Data** — Pandas, pdfplumber, openpyxl
- **Frontend** — HTML, CSS, JavaScript, Chart.js

---

## Project Structure

```
pockit-finance-tracker/
├── finance_model.py    ← Trains ML models on your transaction data
├── app.py              ← Flask API (categorize, predict, import)
├── dashboard.html      ← Frontend dashboard UI
├── keyword_rules.json  ← Auto-generated category keyword rules
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install flask flask-cors pandas scikit-learn pdfplumber openpyxl
```

**2. Train the model**
```bash
python finance_model.py
```

**3. Start the API**
```bash
python app.py
```

**4. Open the dashboard**

Open `dashboard.html` in your browser. The status badge will show **API connected** in green.

**5. Import your bank statement**

Click **↑ Import Bank Statement** and upload your PDF or Excel statement. Pockit will auto-extract and categorize all debit transactions.

---

##  Supported Bank Formats

| Bank | Format |
|---|---|
| IDFC First Bank | PDF, Excel (.xlsx) |
| Most Indian banks | CSV with Date, Particulars, Debit columns |

---

## Dashboard Preview

- **Stats row** — Total spent, avg transaction, impulse buy count
- **Charts** — Spending by category (donut) + monthly trend (line)
- **Insights** — Broke Day, Personality Badge, Subscription Graveyard
- **Roast** — Honest monthly feedback from your own data
- **Fix Panel** — One-click category correction for unknown merchants

---

## How the ML Works

1. **Keyword matching first** — Known merchants (Zomato, Swiggy, Netflix etc.) are caught instantly with 95%+ confidence
2. **Random Forest fallback** — Unknown merchants are classified by the ML model based on amount, time, and day patterns
3. **User corrections** — When you manually fix a category, it's saved to `user_rules.json` and applied to all future imports

---

## Author

**Aarushi Khanna** — [github.com/aarushi0402](https://github.com/aarushi0402)
