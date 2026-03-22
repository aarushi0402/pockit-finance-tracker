from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ── Load everything ────────────────────────────────────────────────────────────
print("Loading models...")
with open('cat_model.pkl','rb') as f: cat_model = pickle.load(f)
with open('spend_model.pkl','rb') as f: spend_model = pickle.load(f)
with open('anomaly_model.pkl','rb') as f: anomaly_model = pickle.load(f)
with open('le_desc.pkl','rb') as f: le_desc = pickle.load(f)
with open('insights.json','r') as f: insights = json.load(f)
with open('keyword_rules.json','r') as f: KEYWORD_RULES = json.load(f)
print("All models loaded!")

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_category_from_keywords(description):
    desc_lower = description.lower().strip()
    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in desc_lower:
                return category
    return 'Uncategorized'

def predict_category(description, amount, hour, day_of_week):
    is_weekend = 1 if day_of_week >= 5 else 0

    # Step 1: keyword matching (fast, accurate for known apps)
    keyword_cat = get_category_from_keywords(description)

    # Step 2: ML model as fallback
    if keyword_cat == 'Uncategorized':
        desc_encoded = le_desc.transform([description])[0] if description in le_desc.classes_ else 0
        features = np.array([[desc_encoded, amount, hour, day_of_week, is_weekend]])
        ml_cat = cat_model.predict(features)[0]
        proba  = cat_model.predict_proba(features)[0]
        confidence = round(float(max(proba)) * 100, 1)
        category = ml_cat
    else:
        category   = keyword_cat
        confidence = 95.0  # keyword match is highly confident

    is_impulse = 1 if (hour >= 22 or hour <= 2) else 0
    is_anomaly = int(anomaly_model.predict([[amount, hour, is_weekend]])[0] == -1)
    return category, confidence, is_impulse, is_anomaly

def clean_description(text):
    text = str(text).strip()
    # Clean UPI format: UPI/DR/refno/MERCHANT/bank/upiid/type
    if 'UPI' in text:
        parts = text.replace('\\', '/').split('/')
        if len(parts) >= 4:
            name = parts[3].strip()
            if name and name.lower() not in ['', 'none', 'upi']:
                return name[:50]
    # Clean other formats
    for prefix in ['NEFT/', 'IMPS/', 'RTGS/', 'ATM/', 'POS/']:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text[:50].strip()

def recalculate_insights():
    df = pd.read_csv('transactions.csv')
    if df.empty:
        return

    total_spent     = round(df['amount'].sum(), 2)
    avg_transaction = round(df['amount'].mean(), 2)
    impulse_df      = df[df['is_impulse'] == 1]
    impulse_total   = round(impulse_df['amount'].sum(), 2)
    impulse_count   = len(impulse_df)
    uncategorized   = len(df[df['category'] == 'Uncategorized'])

    category_breakdown = df.groupby('category')['amount'].sum().round(2).to_dict()
    monthly_trend      = df.groupby('month')['amount'].sum().round(2).to_dict()

    # Subscriptions
    subs = df[df['category'] == 'Subscriptions'].groupby('description').agg(
        total_spent=('amount','sum'), count=('amount','count')
    ).reset_index()
    subs['monthly_avg'] = subs['total_spent'] / max(df['month'].nunique(), 1)
    subscription_list = subs.to_dict('records')

    # Personality
    food_pct = df[df['category']=='Food']['amount'].sum() / total_spent if total_spent else 0
    imp_pct  = impulse_total / total_spent if total_spent else 0
    sub_pct  = df[df['category']=='Subscriptions']['amount'].sum() / total_spent if total_spent else 0
    unc_pct  = df[df['category']=='Uncategorized']['amount'].sum() / total_spent if total_spent else 0

    if imp_pct > 0.3:
        personality      = "Impulsive 🎯"
        personality_desc = "A lot of your spending happens unplanned. Late night purchases are your weakness."
    elif food_pct > 0.4:
        personality      = "Foodie 🍕"
        personality_desc = "Food is clearly your biggest love. Zomato thanks you personally."
    elif sub_pct > 0.15:
        personality      = "Subscription Hoarder 📱"
        personality_desc = "You're subscribed to more things than you probably use. Time for a cleanup."
    elif unc_pct > 0.4:
        personality      = "Social Spender 👥"
        personality_desc = "Most of your money goes to people via UPI. You're either very generous or very social."
    else:
        personality      = "Balanced ⚖️"
        personality_desc = "You spend pretty reasonably across categories. Solid financial habits."

    # Biggest leak
    known_cats = {k:v for k,v in category_breakdown.items() if k != 'Uncategorized'}
    if known_cats:
        biggest_leak        = max(known_cats, key=known_cats.get)
        biggest_leak_amount = round(known_cats[biggest_leak], 2)
    else:
        biggest_leak        = 'N/A'
        biggest_leak_amount = 0

    # Broke day
    days_in_data = max(len(df['date'].unique()), 1)
    avg_daily    = total_spent / days_in_data
    broke_day    = min(int(total_spent / avg_daily), 31) if avg_daily > 0 else 30

    # Roast
    food_spend   = df[df['category']=='Food']['amount'].sum()
    food_pct_n   = (food_spend / total_spent * 100) if total_spent else 0
    zomato_count = len(df[df['description'].str.contains('zomato|swiggy|ZOMATO|SWIGGY', case=False, na=False)])

    if food_pct_n > 35:
        roast = f"You spent {food_pct_n:.0f}% on food. That's {zomato_count} food delivery orders. Your stomach is thriving. Your wallet, not so much."
    elif impulse_total > 3000:
        roast = f"₹{impulse_total:,.0f} on late night purchases. Every ₹200 Zomato order at midnight adds up. Sleep is free."
    elif uncategorized > 20:
        roast = f"{uncategorized} transactions couldn't be categorized — mostly UPI transfers. Either you're very generous or very bad at tracking money."
    elif biggest_leak == 'Rent':
        roast = f"Rent is eating ₹{biggest_leak_amount:,.0f}. That's just life. But everything else better be worth it."
    else:
        roast = f"You made {len(df)} transactions worth ₹{total_spent:,.0f}. Honestly? Could be worse. Could also be better."

    # Future warning
    monthly_avg      = df.groupby('month')['amount'].sum().mean() if not df.empty else 0
    future_6_months  = round(monthly_avg * 6, 2)
    future_12_months = round(monthly_avg * 12, 2)

    insights.update({
        'total_spent':            total_spent,
        'avg_transaction':        avg_transaction,
        'category_breakdown':     category_breakdown,
        'monthly_trend':          monthly_trend,
        'impulse_total':          impulse_total,
        'impulse_count':          impulse_count,
        'uncategorized_count':    uncategorized,
        'personality':            personality,
        'personality_desc':       personality_desc,
        'biggest_leak':           biggest_leak,
        'biggest_leak_amount':    biggest_leak_amount,
        'subscription_list':      subscription_list,
        'roast':                  roast,
        'future_6_months':        future_6_months,
        'future_12_months':       future_12_months,
        'broke_day':              broke_day,
    })
    with open('insights.json','w') as f:
        json.dump(insights, f)

def find_headers(row_lower):
    date_idx  = next((i for i,c in enumerate(row_lower) if 'transaction' in c and 'date' in c), None)
    if date_idx is None:
        date_idx = next((i for i,c in enumerate(row_lower) if c.strip() == 'date'), None)
    desc_idx  = next((i for i,c in enumerate(row_lower) if any(x in c for x in ['particular','description','narration','details'])), None)
    debit_idx = next((i for i,c in enumerate(row_lower) if c.strip() in ['debit','debit amount','withdrawal','dr']), None)
    return date_idx, desc_idx, debit_idx

def parse_date(date_val):
    for fmt in ['%d-%b-%Y','%d-%m-%Y','%Y-%m-%d','%d/%m/%Y','%m/%d/%Y','%b %d %Y']:
        try:
            return datetime.strptime(date_val.strip(), fmt)
        except:
            continue
    return None

def process_rows(rows, date_idx, desc_idx, debit_idx):
    added, skipped = 0, 0
    df_existing = pd.read_csv('transactions.csv')
    new_rows = []

    for row in rows:
        try:
            if not row or all(str(c).strip() in ['','None','nan'] for c in row):
                continue

            date_val  = str(row[date_idx]).strip() if row[date_idx] else ''
            if not date_val or date_val.lower() in ['none','nan','']:
                skipped += 1; continue

            debit_val = str(row[debit_idx]).replace(',','').strip() if debit_idx is not None and debit_idx < len(row) and row[debit_idx] else ''
            if not debit_val or debit_val.lower() in ['none','nan','']:
                skipped += 1; continue

            amount = float(debit_val)
            if amount <= 0:
                skipped += 1; continue

            raw_desc    = str(row[desc_idx]).strip() if desc_idx is not None and desc_idx < len(row) and row[desc_idx] else 'Unknown'
            description = clean_description(raw_desc)

            date = parse_date(date_val)
            if not date:
                skipped += 1; continue

            hour        = 12
            day_of_week = date.weekday()
            is_weekend  = 1 if day_of_week >= 5 else 0
            month       = date.month

            category, confidence, is_impulse, is_anomaly = predict_category(
                description, amount, hour, day_of_week
            )

            new_rows.append({
                'date':        date.strftime('%Y-%m-%d'),
                'description': description,
                'amount':      round(amount, 2),
                'category':    category,
                'hour':        hour,
                'day_of_week': day_of_week,
                'is_weekend':  is_weekend,
                'is_impulse':  is_impulse,
                'month':       month
            })
            added += 1
        except:
            skipped += 1
            continue

    if new_rows:
        new_df   = pd.DataFrame(new_rows)
        combined = pd.concat([df_existing, new_df], ignore_index=True)
        combined.to_csv('transactions.csv', index=False)
        recalculate_insights()

    return added, skipped

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/insights', methods=['GET'])
def get_insights():
    return jsonify(insights)

@app.route('/categorize', methods=['POST'])
def categorize():
    try:
        data        = request.json
        description = data.get('description','')
        amount      = float(data.get('amount', 0))
        hour        = int(data.get('hour', 12))
        day_of_week = int(data.get('day_of_week', 0))
        category, confidence, is_impulse, is_anomaly = predict_category(description, amount, hour, day_of_week)
        return jsonify({'category': category, 'confidence': confidence,
                        'is_impulse': bool(is_impulse), 'is_anomaly': bool(is_anomaly)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/add-transaction', methods=['POST'])
def add_transaction():
    try:
        data        = request.json
        description = data.get('description','')
        amount      = float(data.get('amount', 0))
        date_str    = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        hour        = int(data.get('hour', 12))
        date        = datetime.strptime(date_str, '%Y-%m-%d')
        day_of_week = date.weekday()

        category, confidence, is_impulse, is_anomaly = predict_category(
            description, amount, hour, day_of_week
        )

        new_row = {
            'date': date_str, 'description': description, 'amount': amount,
            'category': category, 'hour': hour, 'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_impulse': is_impulse, 'month': date.month
        }
        df     = pd.read_csv('transactions.csv')
        df     = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv('transactions.csv', index=False)
        recalculate_insights()

        return jsonify({'success': True, 'category': category,
                        'confidence': confidence, 'is_impulse': bool(is_impulse),
                        'message': f'Added as {category}!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/update-category', methods=['POST'])
def update_category():
    """Let user manually fix a wrong category — teaches the system"""
    try:
        data        = request.json
        description = data.get('description','')
        category    = data.get('category','')

        df = pd.read_csv('transactions.csv')
        df.loc[df['description'] == description, 'category'] = category
        df.to_csv('transactions.csv', index=False)
        recalculate_insights()

        # Also save to user rules
        user_rules = {}
        try:
            with open('user_rules.json','r') as f:
                user_rules = json.load(f)
        except:
            pass
        user_rules[description.lower()] = category
        with open('user_rules.json','w') as f:
            json.dump(user_rules, f)

        return jsonify({'success': True, 'message': f'Updated {description} → {category}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload-statement', methods=['POST'])
def upload_statement():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file       = request.files['file']
        filename   = file.filename.lower()
        file_bytes = file.read()

        rows = []
        date_idx = desc_idx = debit_idx = None

        # ── PDF ───────────────────────────────────────────────────────────────
        if filename.endswith('.pdf'):
            import pdfplumber
            from io import BytesIO
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    table = page.extract_table()
                    if not table: continue
                    for row in table:
                        if not row or all(c is None for c in row): continue
                        if date_idx is None:
                            row_lower = [str(c).lower().replace('\n',' ').strip() if c else '' for c in row]
                            d, de, db = find_headers(row_lower)
                            if d is not None:
                                date_idx, desc_idx, debit_idx = d, de, db
                                continue
                        if date_idx is not None:
                            rows.append([str(c).replace('\n',' ').strip() if c else '' for c in row])

        # ── XLSX ──────────────────────────────────────────────────────────────
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            import openpyxl
            from io import BytesIO
            wb = openpyxl.load_workbook(BytesIO(file_bytes))
            ws = wb.active
            for row in ws.iter_rows(values_only=True):
                if not row: continue
                if date_idx is None:
                    row_lower = [str(c).lower().strip() if c else '' for c in row]
                    d, de, db = find_headers(row_lower)
                    if d is not None:
                        date_idx, desc_idx, debit_idx = d, de, db
                        continue
                if date_idx is not None:
                    rows.append(list(row))

        # ── CSV ───────────────────────────────────────────────────────────────
        else:
            content = file_bytes.decode('utf-8', errors='ignore')
            for line in content.splitlines():
                cols = line.split(',')
                if date_idx is None:
                    lower = [c.lower().strip() for c in cols]
                    d, de, db = find_headers(lower)
                    if d is not None:
                        date_idx, desc_idx, debit_idx = d, de, db
                        continue
                if date_idx is not None:
                    rows.append(cols)

        if date_idx is None:
            return jsonify({'error': 'Could not find transaction headers. Make sure your file has a Transaction Date column.'}), 400

        added, skipped = process_rows(rows, date_idx, desc_idx, debit_idx)
        return jsonify({'success': True, 'added': added, 'skipped': skipped,
                        'message': f'Imported {added} transactions! ({skipped} skipped)'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/transactions', methods=['GET'])
def get_transactions():
    try:
        df     = pd.read_csv('transactions.csv')
        recent = df.sort_values('date', ascending=False).head(50)
        return jsonify(recent.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/uncategorized', methods=['GET'])
def get_uncategorized():
    try:
        df  = pd.read_csv('transactions.csv')
        unc = df[df['category'] == 'Uncategorized'][['description','amount','date']].drop_duplicates('description').head(20)
        return jsonify(unc.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'accuracy': insights.get('categorization_accuracy', 0)})

if __name__ == '__main__':
    print("\n✓ Pockit API running at http://localhost:5000")
    print("✓ Open dashboard.html in your browser\n")
    app.run(debug=True, port=5000)