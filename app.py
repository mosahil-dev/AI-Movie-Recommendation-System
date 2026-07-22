from flask import Flask, render_template, request
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Load Data
movies = pickle.load(open("movies.pkl", "rb"))
if not os.path.exists("similarity.pkl"):
    raise FileNotFoundError(
        "similarity.pkl not found.\n"
        "Please run: python build_similarity.py"
    )
similarity = pickle.load(open("similarity.pkl", "rb"))

import os
print("Current Folder:" , os.getcwd())
print("Movies Columns:" , movies.columns.tolist())

# Create lowercase title column once
movies["Title_lower"] = movies["Title"].str.lower()


def recommend(movie_name):
    movie_name = movie_name.lower()

    if movie_name not in movies["Title_lower"].values:
        return None

    index = movies[movies["Title_lower"] == movie_name].index[0]

    distances = list(enumerate(similarity[index]))

    movie_list = sorted(
        distances,
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    recommendations = []

    for i in movie_list:

        movie = movies.iloc[i[0]]

        print(movie.index.tolist())

        recommendations.append({
            "title": movie["Title"],
            "genre": movie["Genre"],
            "rating": movie["IMDb Score"],
            "summary": movie["Summary"],
            "poster": movie["Poster"],
            "imdb": movie["IMDb Link"],
            "netflix": movie["Netflix Link"],
            "release": movie["Release Date"]
        })

    return recommendations
@app.route("/")
def home():

    movie_list = sorted(movies["Title"].dropna().unique())

    return render_template(
        "index.html",
        movies=movie_list
    )


@app.route("/recommend", methods=["POST"])
def predict():

    movie = request.form["movie"]

    recommendations = recommend(movie)

    return render_template(
        "result.html",
        movie=movie,
        recommendations=recommendations
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT" , 5000))
    app.run(host="0.0.0.0",port=port)