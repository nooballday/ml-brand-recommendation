import pandas as pd
import random

# Load brands
brands = pd.read_csv("brands.csv")
brand_ids = brands['brand_id'].tolist()

categories = ['Streetwear', 'Luxury', 'Sportswear', 'Casual', 'Formal']
price_ranges = ['Low', 'Medium', 'High']
genders = ['Men', 'Women', 'Unisex']

NUM_USERS = 50

# Generate user profiles
def generate_user(i):
    return {
        'user_id': f'user_{i:03}',
        'preferred_categories': random.sample(categories, k=2),
        'preferred_price_range': random.choice(price_ranges),
        'preferred_gender_fit': random.choice(genders),
        'sustainability_focused': random.choice([True, False])
    }

users = pd.DataFrame([generate_user(i) for i in range(NUM_USERS)])
users.to_csv("users.csv", index=False)

# Simulate ratings with natural bias
def generate_rating():
    return random.choices([1, 2, 3, 4, 5], weights=[1, 2, 4, 5, 3])[0]

interactions = []
for _, user in users.iterrows():
    rated_brands = random.sample(brand_ids, k=random.randint(30, 60))
    for brand_id in rated_brands:
        interactions.append({
            'user_id': user['user_id'],
            'brand_id': brand_id,
            'rating': generate_rating()
        })

ratings_df = pd.DataFrame(interactions)
ratings_df.to_csv("user_brand_ratings.csv", index=False)

print("✅ More realistic user-brand ratings generated.")

