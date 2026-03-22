import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import json

# ── Keyword-based category rules (works for ANY user) ─────────────────────────
KEYWORD_RULES = {
    'Food': [
        'zomato','swiggy','dominos','mcdonalds','mcdonald','kfc','pizza','burger',
        'starbucks','dunkin','subway','blinkit','zepto','bigbasket','grofers',
        'canteen','mess','cafe','restaurant','food','snack','chai','tea','juice',
        'nescafe','haldiram','amul','eat','kitchen','dhaba','hotel','bakery',
        'grocery','vegetable','fruit','milk','dairy'
    ],
    'Transport': [
        'ola','uber','rapido','redbus','irctc','railway','metro','bus','auto',
        'petrol','diesel','fuel','parking','toll','makemytrip','goibibo','yatra',
        'indigo','spicejet','airindia','flight','cab','taxi','swift','fastag'
    ],
    'Shopping': [
        'amazon','flipkart','myntra','meesho','ajio','nykaa','tatacliq','snapdeal',
        'reliance','dmart','bigbazar','shopify','retail','shop','store','mart',
        'decathlon','puma','nike','adidas','zara','h&m','lifestyle','westside',
        'jabong','lenskart','firstcry','babycare','clothing','fashion'
    ],
    'Entertainment': [
        'netflix','hotstar','prime video','sony liv','zee5','mx player','voot',
        'bookmyshow','pvr','inox','cinepolis','gaming','steam','playstation',
        'spotify','gaana','jiosaavn','wynk','youtube','ticketnew','zomato live',
        'paytm insider','district','movie','cinema','concert','event','show'
    ],
    'Subscriptions': [
        'netflix','spotify','amazon prime','hotstar','apple','google','microsoft',
        'adobe','dropbox','notion','slack','zoom','icloud','playstore','appstore',
        'linkedin','coursera','udemy','subscription','monthly','annual','renewal',
        'insurance','lic','policy','premium'
    ],
    'Rent': [
        'rent','pg','hostel','landlord','house','flat','apartment','society',
        'maintenance','electricity','water','gas','bill','broadband','internet',
        'wifi','airtel','jio','vodafone','bsnl','tata sky','dish tv','recharge'
    ],
    'Education': [
        'udemy','coursera','unacademy','byju','vedantu','khan','college','school',
        'university','institute','tuition','coaching','exam','fee','books','library',
        'stationery','notebook','pen','pencil','iit','nit','study','course','class',
        'edutech','skill','learning','certificate','degree'
    ],
    'Health': [
        'pharmacy','medical','doctor','hospital','clinic','apollo','medplus',
        'netmeds','1mg','practo','gym','fitness','cult','healthify','yoga',
        'medicine','tablet','injection','test','lab','diagnostic','health',
        'wellness','pharma','chemist','dentist','optician','lens','spectacles'
    ],
}

def get_category_from_keywords(description):
    desc_lower = description.lower().strip()
    for category, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw in desc_lower:
                return category
    return 'Uncategorized'

# ── Build training data ────────────────────────────────────────────────────────
print("Building training data...")

training_samples = []

for category, keywords in KEYWORD_RULES.items():
    for kw in keywords[:8]:  # use first 8 keywords per category
        training_samples.append({
            'description': kw,
            'amount': np.random.uniform(50, 3000),
            'category': category,
            'hour': np.random.randint(8, 22),
            'day_of_week': np.random.randint(0, 7),
        })

# Add more realistic samples
extra_samples = [
    # Food
    ('zomato order', 350, 'Food', 20, 1),
    ('swiggy delivery', 280, 'Food', 19, 0),
    ('dominos pizza', 450, 'Food', 21, 6),
    ('mcdonalds', 200, 'Food', 13, 5),
    ('blinkit groceries', 800, 'Food', 18, 2),
    ('zepto delivery', 500, 'Food', 17, 3),
    ('canteen food', 80, 'Food', 13, 1),
    ('mess bill', 2500, 'Food', 12, 0),
    ('chai stall', 20, 'Food', 8, 2),
    ('restaurant dinner', 600, 'Food', 20, 6),
    # Transport
    ('ola cab', 120, 'Transport', 9, 0),
    ('uber ride', 200, 'Transport', 14, 2),
    ('rapido bike', 60, 'Transport', 8, 1),
    ('metro recharge', 300, 'Transport', 7, 0),
    ('petrol pump', 1000, 'Transport', 11, 5),
    ('irctc train', 800, 'Transport', 15, 3),
    ('bus pass', 500, 'Transport', 8, 0),
    ('makemytrip flight', 5000, 'Transport', 10, 6),
    # Shopping
    ('amazon purchase', 1500, 'Shopping', 15, 6),
    ('flipkart order', 2000, 'Shopping', 16, 5),
    ('myntra clothes', 1200, 'Shopping', 20, 0),
    ('nykaa beauty', 800, 'Shopping', 21, 1),
    ('meesho order', 600, 'Shopping', 19, 2),
    ('decathlon sports', 2500, 'Shopping', 12, 6),
    # Entertainment
    ('netflix subscription', 499, 'Entertainment', 0, 5),
    ('bookmyshow movie', 300, 'Entertainment', 18, 6),
    ('pvr cinema', 400, 'Entertainment', 17, 5),
    ('spotify music', 119, 'Entertainment', 0, 0),
    ('gaming steam', 500, 'Entertainment', 22, 6),
    # Subscriptions
    ('amazon prime', 299, 'Subscriptions', 0, 2),
    ('hotstar subscription', 899, 'Subscriptions', 0, 1),
    ('apple icloud', 75, 'Subscriptions', 0, 3),
    ('google storage', 130, 'Subscriptions', 0, 0),
    ('linkedin premium', 1600, 'Subscriptions', 0, 4),
    # Rent
    ('pg rent payment', 8000, 'Rent', 1, 1),
    ('house rent', 12000, 'Rent', 1, 0),
    ('electricity bill', 500, 'Rent', 10, 2),
    ('airtel broadband', 999, 'Rent', 9, 0),
    ('jio recharge', 239, 'Rent', 12, 3),
    ('water bill', 200, 'Rent', 11, 1),
    # Education
    ('udemy course', 999, 'Education', 14, 2),
    ('coursera certificate', 3000, 'Education', 15, 1),
    ('college fee', 50000, 'Education', 10, 0),
    ('books stationery', 400, 'Education', 11, 3),
    ('coaching fee', 5000, 'Education', 9, 1),
    # Health
    ('pharmacy medicine', 150, 'Health', 12, 2),
    ('gym membership', 800, 'Health', 7, 0),
    ('doctor consultation', 500, 'Health', 11, 3),
    ('apollo pharmacy', 300, 'Health', 13, 1),
    ('1mg medicine', 200, 'Health', 14, 2),
    # Uncategorized (personal UPI transfers)
    ('personal transfer', 500, 'Uncategorized', 14, 0),
    ('upi payment', 200, 'Uncategorized', 15, 1),
    ('sent money', 1000, 'Uncategorized', 16, 2),
    ('friend payment', 300, 'Uncategorized', 17, 3),
]

for desc, amt, cat, hr, dow in extra_samples:
    training_samples.append({
        'description': desc, 'amount': amt, 'category': cat,
        'hour': hr, 'day_of_week': dow,
    })

df = pd.DataFrame(training_samples)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_impulse'] = ((df['hour'] >= 22) | (df['hour'] <= 2)).astype(int)

print(f"Training samples: {len(df)}")
print("Category distribution:")
print(df['category'].value_counts())

# ── Train model ────────────────────────────────────────────────────────────────
print("\nTraining model...")
le_desc = LabelEncoder()
df['desc_encoded'] = le_desc.fit_transform(df['description'])

features = ['desc_encoded', 'amount', 'hour', 'day_of_week', 'is_weekend']
X = df[features]
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cat_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
cat_model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, cat_model.predict(X_test))
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, cat_model.predict(X_test)))

# ── Anomaly model ──────────────────────────────────────────────────────────────
anomaly_model = IsolationForest(contamination=0.05, random_state=42)
anomaly_model.fit(df[['amount', 'hour', 'is_weekend']])

spend_model = LinearRegression()
spend_model.fit([[1,0,0],[2,0,0],[3,0,0]], [0,0,0])

# ── Save models ────────────────────────────────────────────────────────────────
print("\nSaving models...")
with open('cat_model.pkl','wb') as f: pickle.dump(cat_model, f)
with open('spend_model.pkl','wb') as f: pickle.dump(spend_model, f)
with open('anomaly_model.pkl','wb') as f: pickle.dump(anomaly_model, f)
with open('le_desc.pkl','wb') as f: pickle.dump(le_desc, f)
with open('keyword_rules.json','w') as f: json.dump(KEYWORD_RULES, f)

# ── Empty transactions + insights ─────────────────────────────────────────────
empty_df = pd.DataFrame(columns=[
    'date','description','amount','category',
    'hour','day_of_week','is_weekend','is_impulse','month'
])
empty_df.to_csv('transactions.csv', index=False)

insights = {
    'total_spent': 0, 'avg_transaction': 0,
    'categorization_accuracy': round(accuracy, 4),
    'category_breakdown': {}, 'monthly_trend': {},
    'broke_day': 0, 'impulse_total': 0, 'impulse_count': 0,
    'personality': 'No Data Yet',
    'personality_desc': 'Upload your bank statement to get started!',
    'biggest_leak': 'N/A', 'biggest_leak_amount': 0,
    'subscription_list': [], 'uncategorized_count': 0,
    'roast': 'Upload your statement to get roasted. 🔥',
    'future_6_months': 0, 'future_12_months': 0, 'anomaly_count': 0
}
with open('insights.json','w') as f: json.dump(insights, f)

print("\n✓ All models saved!")
print("✓ Empty transactions.csv created")
print("✓ Keyword rules saved")
print(f"✓ Accuracy: {accuracy*100:.1f}%")
print("\nNow run: python app.py")