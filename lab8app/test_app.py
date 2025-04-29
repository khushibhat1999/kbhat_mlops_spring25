import requests

# comment = {'reddit_comment':'Testing a comment.'}

# #comment = {'reddit_comment':'hey Reddit user, I have some snake oil to sell you!'}

# url = 'http://127.0.0.1:8000/predict'
# response = requests.post(url, json=comment)
# print(response.json())

sample_input = {
    "acousticness": 0.514,
    "danceability": 0.735,
    "duration_ms": 201000,
    "energy": 0.812,
    "instrumentalness": 0.0,
    "key": 5,
    "liveness": 0.0897,
    "loudness": -4.554,
    "mode": 1,
    "speechiness": 0.0454,
    "tempo": 120.018,
    "valence": 0.624
}

url = 'http://127.0.0.1:8000/predict'
response = requests.post(url, json=sample_input)

print(response.json())