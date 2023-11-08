"""
Movie recommendations algorithm

Authors: Dariusz Karasiewicz, Mikołaj Kusiński
"""

import argparse
import json
import numpy as np
from compute_scores import pearson_score
def build_arg_parser():
    """
    This function defines an argument parser which takes classifier type as an input parameter.

    return:
    ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Find recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser

def get_recommendations(dataset, input_user):
    """
    This function get the movie recommendations for given user.

    input:
    Any dataset
    Any input_user

    return:
    list movie_recommendations
    """
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)
        if similarity_score <= 0:
            continue

        filtered_list = [x for x in dataset[user] if x not in dataset[input_user] or dataset[input_user][x] == 0]

        for item in filtered_list:
            if item in overall_scores:
                overall_scores[item] += dataset[user][item] * similarity_score
            else:
                overall_scores[item] = dataset[user][item] * similarity_score

            if item in similarity_scores:
                similarity_scores[item] += similarity_score
            else:
                similarity_scores[item] = similarity_score

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()])

    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    movie_recommendations = [movie for _, movie in movie_scores]
    return movie_recommendations

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    ratings_file = 'data.json'
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    print("\nMovie recommendations for " + user + ":")
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '. ' + movie)

    for i, movie in enumerate(movies[:3]):
        print(str(i + 1) + '. ' + movie)
    print("")
    print("Movie  antirecommendations" + user + ":")
    for i, movie in enumerate(movies[-3:]):
        print(str(i + 1) + '. ' + movie)
