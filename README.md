## Dependencies
`fastapi` `pandas` `scikit-learn` `uvicorn` `scipy` `numpy`

## How to start

python3 generate_brand_csv.py


python3 generate_user_csv.py

## How to run

uvicorn main:app --reload

### swagger

http://localhost:8000/docs#/default/get_recommendations_recommend__user_id__get

### Example request
http://localhost:8000/recommend/user_07?n=10&k=5&alpha=0.7&min_similarity=0.05
