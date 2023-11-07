import argparse
import json
import numpy as np
from compute_scores import pearson_score
# from collaborative_filtering import find_similar_users
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser

# Get movie recommendations for the input user
def get_recommendations(dataset, input_user):
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

    # Generate movie ranks by normalization
    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()])

    # Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations
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
    # for i, movie in enumerate(movies):
    #     print(str(i + 1) + '. ' + movie)

    for i, movie in enumerate(movies[:3]):
        print(str(i + 1) + '. ' + movie)
    print("")
    print("Movie  antirecommendations" + user + ":")
    for i, movie in enumerate(movies[-3:]):
        print(str(i + 1) + '. ' + movie)
