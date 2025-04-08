import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

spotify_df = pd.read_csv("songs_normalize.csv")

y = spotify_df['popularity']
X=spotify_df.drop(['popularity', 'artist', 'song', 'genre'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

pd.DataFrame(X_train).to_csv("X_train_lab3.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test_lab3.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train_lab3.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test_lab3.csv", index=False)

# Save pipeline
with open('pipeline.pkl','wb') as f:
    pickle.dump(model,f)


