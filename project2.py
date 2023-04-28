from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import json
import argparse

# returns yummly.json as a dataframe
def read_data():
    with open('docs/yummly.json') as f:
        data = json.load(f)
    dataframe = pd.DataFrame(data)
    return dataframe

def print_json(cuisine, cuisine_score, n_closest):
    print_json = {}
    closest_list = []

    for dict in n_closest:
        closest_list.append({'id': float(dict[0]), 'score': float(round(dict[1],2))})

    print_json = {
        'cuisine': cuisine,
        'score': float(round(cuisine_score,2)),
        'n_closest': closest_list
    }
    ret = json.loads(json.dumps(print_json))
    json_format = json.dumps(ret, indent=4)
    return json_format

# create a corpus of ingredientss
def get_corpus(df, args):
    """
    vectorize the ingredients
    df: dataframe of the input file (yummly.json)
    args: ingrediants arguments from the command line
    """
    list_of_ingredients = df['ingredients'].tolist()
    args_ingrediants = args
    corpus = []

    #go throught the list of ingredients and add them to the corpus
    for i in list_of_ingredients:
        corpus.append(' '.join(i))
    
    #add the command line ingredients to the corpus
    corpus.append(' '.join(args_ingrediants))

    return corpus

def predict_cuisine(df, corpus, N):
    
    # get the cuisine labels
    cuisine_labels = df['cuisine']

    # vectorize the ingredients corpus
    vectorizer = TfidfVectorizer()
    ingredients_fit = vectorizer.fit_transform(corpus)
    # get everything but our command line ingredients
    ingredients = ingredients_fit[:-1]

    # get the command line ingredients
    command_line_ingredients = ingredients_fit[-1]

    # create a trainer
    X_train, X_test, y_train, y_test = train_test_split(ingredients, cuisine_labels, test_size=0.2, random_state=42)
    # create a knn model
    knn = KNeighborsClassifier(n_neighbors=N)
    # fit the model
    knn.fit(X_train, y_train)
    predicted_cuisine = knn.predict(command_line_ingredients)

    #get the max cuisine score
    max_score = np.max(knn.predict_proba(command_line_ingredients))

    return predicted_cuisine[0], max_score, ingredients, command_line_ingredients

def predict_n_closest(command_line_ingredients, ingredients, N):

     # get the N closest recipes
    n_scores = cosine_similarity(command_line_ingredients, ingredients)
    sorted_n_scores = n_scores[0].argsort()[-N:][::-1]

    # get the N closest recipes ids
    n_closes_recipes_ids = [(i, n_scores[0][i]) for i in sorted_n_scores]

    return n_closes_recipes_ids

if __name__ == "__main__":
    #pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies"
    parser = argparse.ArgumentParser(description='Project 2')
    parser.add_argument('--ingredient', type=str, required=True, action='append', help='ingredient')
    parser.add_argument('--N', type=int, required=True, help='N')
    args = parser.parse_args()

    # Load data
    df = read_data()


    corpus = get_corpus(df, args.ingredient)
    cuisine, cuisine_score, ingredients, input_ingredients = predict_cuisine(df, corpus, args.N)
    n_closest = predict_n_closest(input_ingredients, ingredients, args.N)

    # Print results
    print(print_json(cuisine, cuisine_score, n_closest))