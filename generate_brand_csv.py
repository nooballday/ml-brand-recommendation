import pandas as pd
import numpy as np
import random

categories = ['Streetwear', 'Luxury', 'Sportswear', 'Casual', 'Formal']
price_ranges = ['Low', 'Medium', 'High']
genders = ['Men', 'Women', 'Unisex']
countries = ['US', 'UK', 'JP', 'FR', 'IT', 'KR', 'CN']

def generate_brand(i):
    return {
        'brand_id': f'brand_{i:04}',
        'brand_name': f'Brand{i}',
        'category': random.choice(categories),
        'price_range': random.choice(price_ranges),
        'gender_fit': random.choice(genders),
        'sustainable': random.choice([True, False]),
        'popularity': round(random.uniform(0, 1), 2),
        'country': random.choice(countries)
    }

brands = pd.DataFrame([generate_brand(i) for i in range(1000)])
brands.to_csv("brands.csv", index=False)
print("✅ Brands generated")

