import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data/grocery.csv")

# Encode categorical columns
le_item = LabelEncoder()
le_season = LabelEncoder()
le_demand = LabelEncoder()

data['item'] = le_item.fit_transform(data['item'])
data['season'] = le_season.fit_transform(data['season'])
data['demand'] = le_demand.fit_transform(data['demand'])

# Features (X) and Target (y)
X = data[['item', 'season', 'quantity', 'demand']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

print("✅ Model trained successfully!\n")

# Take user input
item = input("Enter item (tomato/potato/onion/apple/banana): ")
season = input("Enter season (summer/winter/all): ")
quantity = int(input("Enter quantity: "))
demand = input("Enter demand (low/medium/high): ")

# Convert input into numbers
try:
    item_encoded = le_item.transform([item])[0]
    season_encoded = le_season.transform([season])[0]
    demand_encoded = le_demand.transform([demand])[0]

    # Predict price
    prediction = model.predict([[item_encoded, season_encoded, quantity, demand_encoded]])

    print("\n💰 Predicted Price:", round(prediction[0], 2))

except:
    print("\n❌ Invalid input! Please enter values from dataset.")