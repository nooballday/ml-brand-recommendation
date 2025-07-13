import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# Load brands
brands = pd.read_csv("brands.csv")

# Load users
users = pd.read_csv("users.csv")

# Load ratings
ratings = pd.read_csv("user_brand_ratings.csv")

# Create user-item matrix (pivot table)
user_brand_matrix = ratings.pivot_table(
    index='user_id', columns='brand_id', values='rating'
).fillna(0)

# Convert to sparse matrix
sparse_matrix = csr_matrix(user_brand_matrix.values)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(sparse_matrix)

print("✅ KNN model ready")

brand_features = pd.get_dummies(
    brands[['brand_id', 'category', 'price_range', 'gender_fit']],
    columns=['category', 'price_range', 'gender_fit']
)

brand_features['sustainable'] = brands['sustainable'].astype(int)

scaler = MinMaxScaler()
brand_features['popularity'] = scaler.fit_transform(brands[['popularity']])

brand_features = brand_features.set_index('brand_id')

def recommend_brands_for_user(user_id, k=5, n_recommendations=5, min_similarity=0.01, min_raters=2):
    if user_id not in user_brand_matrix.index:
        print("User not found.")
        return []

    # Step 1: Find similar users using KNN
    user_idx = user_brand_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors([user_brand_matrix.iloc[user_idx]], n_neighbors=k+1)

    # Exclude self
    similar_user_indices = indices.flatten()[1:]
    similar_users = user_brand_matrix.index[similar_user_indices]
    similarities = 1 - distances.flatten()[1:]

    # Build DataFrame of similar users
    sim_df = pd.DataFrame({
        'user_id': similar_users,
        'similarity': similarities
    })

    # Filter by minimum similarity
    sim_df = sim_df[sim_df['similarity'] >= min_similarity]
    if sim_df.empty:
        print("No sufficiently similar users found.")
        return []

    # Step 2: Collect ratings from similar users
    candidate_brands = ratings[ratings['user_id'].isin(sim_df['user_id'])]
    rated_by_user = set(ratings[ratings['user_id'] == user_id]['brand_id'])

    # Step 3: Filter out brands already rated by the current user
    candidates = candidate_brands[~candidate_brands['brand_id'].isin(rated_by_user)].copy()

    # Add similarity column
    candidates['similarity'] = candidates['user_id'].map(sim_df.set_index('user_id')['similarity'])

    # Step 4: Calculate weighted average ratings
    weighted_scores = (
        candidates
        .groupby('brand_id', group_keys=False)
        .apply(lambda df: (df['rating'] * df['similarity']).sum() / df['similarity'].sum())
        .reset_index(name='weighted_rating')
    )

    # Step 5: Add average similarity per brand
    avg_sim = (
        candidates
        .groupby('brand_id')['similarity']
        .mean()
        .reset_index(name='avg_similarity')
    )

    # Step 6: Add number of raters per brand
    num_raters = (
        candidates
        .groupby('brand_id')['user_id']
        .nunique()
        .reset_index(name='num_raters')
    )

    # Merge results
    result = weighted_scores.merge(avg_sim, on='brand_id')
    result = result.merge(num_raters, on='brand_id')
    result = result[result['num_raters'] >= min_raters]

    # Join with brand metadata
    result = result.merge(brands, on='brand_id', how='left')

    # Sort by weighted rating
    result = result.sort_values(by='weighted_rating', ascending=False).head(n_recommendations)

    return result.to_dict(orient='records')

recs = recommend_brands_for_user("user_01", k=7, n_recommendations=5)
for r in recs:
    print(r)



# Start of content based
def get_user_preference_vector(user_id):
    user = users[users['user_id'] == user_id]
    if user.empty:
        print("User not found.")
        return None

    user = user.iloc[0]

    # Create a zero-filled DataFrame with brand_features columns
    user_vector = pd.DataFrame([0] * len(brand_features.columns), index=brand_features.columns).T

    # Match one-hot columns
    for cat in user['preferred_categories'].strip("[]").replace("'", "").split(', '):
        col = f'category_{cat}'
        if col in user_vector.columns:
            user_vector.at[0, col] = 1

    price_col = f"price_range_{user['preferred_price_range']}"
    if price_col in user_vector.columns:
        user_vector.at[0, price_col] = 1

    gender_col = f"gender_fit_{user['preferred_gender_fit']}"
    if gender_col in user_vector.columns:
        user_vector.at[0, gender_col] = 1

    user_vector.at[0, 'sustainable'] = int(user['sustainability_focused'])

    # Popularity preference is not explicit — we could set to mean or skip
    user_vector.at[0, 'popularity'] = 0.5  # neutral

    return user_vector

def get_content_scores(user_vector):
    # Compute cosine similarity between user vector and all brand feature vectors
    scores = cosine_similarity(user_vector.values, brand_features.values)[0]

    # Map back to brand IDs
    content_scores = pd.DataFrame({
        'brand_id': brand_features.index,
        'content_score': scores
    })

    return content_scores

user_vector = get_user_preference_vector("user_01")
content_scores = get_content_scores(user_vector)
print(content_scores.sort_values(by="content_score", ascending=False).head())

def recommend_hybrid_for_user(user_id, k=10, n_recommendations=5, min_similarity=0.05, alpha=0.7):
    # 1. Get KNN-based recommendations
    knn_results = recommend_brands_for_user(user_id, k=k, n_recommendations=100, min_similarity=min_similarity, min_raters=1)

    # Convert to DataFrame
    knn_df = pd.DataFrame(knn_results)
    knn_df['normalized_knn_score'] = knn_df['weighted_rating'] / 5.0

    # 2. Get content-based similarity scores
    user_vector = get_user_preference_vector(user_id)
    if user_vector is None:
        return []

    content_scores = get_content_scores(user_vector)

    # 3. Merge both
    combined = pd.merge(content_scores, knn_df, on='brand_id', how='outer')

    # Fill missing values
    combined['normalized_knn_score'] = combined['normalized_knn_score'].fillna(0)
    combined['content_score'] = combined['content_score'].fillna(0)

    # 4. Blend score
    combined['final_score'] = alpha * combined['normalized_knn_score'] + (1 - alpha) * combined['content_score']

    # 5. Merge brand metadata
    combined = combined.merge(brands, on='brand_id', how='left')

    # 6. Return top N recommendations
    result = combined.sort_values(by='final_score', ascending=False).head(n_recommendations)

    return result.to_dict(orient='records')

recs = recommend_hybrid_for_user("user_01", k=9, n_recommendations=5, alpha=0.7)
for r in recs:
    print(r)
