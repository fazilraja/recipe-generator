from project2 import get_corpus, read_data, predict_cuisine, predict_n_closest
import pandas as pd

df = read_data()

def test_get_input():
    #check the type
    assert type(df) == pd.core.frame.DataFrame

def test_get_corpus():
    # get the corpus
    corpus = get_corpus(df, ["paprika", "banana", "rice krispies"])
    # check the type
    assert type(corpus) == list
    # check the length
    assert len(corpus) == 39775
    # check the first element
    assert corpus[0] == 'romaine lettuce black olives grape tomatoes garlic pepper purple onion seasoning garbanzo beans feta cheese crumbles'
    # check the last element
    assert corpus[-1] == 'paprika banana rice krispies'

def test_predict_cuisine():
    # get the corpus
    corpus = get_corpus(df, ["paprika", "banana", "rice krispies"])
    # predict the cuisine
    cuisine, cuisine_score, ingredients, input_ingredients = predict_cuisine(df, corpus, 5)
    # check the type
    assert type(cuisine) == str
    # check the value
    assert cuisine == 'vietnamese'

    # check the value
    assert cuisine_score == 0.6

def predict_n_closest():
    # get the corpus
    corpus = get_corpus(df, ["paprika", "banana", "rice krispies"])
    # predict the cuisine
    cuisine, cuisine_score, ingredients, input_ingredients = predict_cuisine(df, corpus, 5)
    # predict the n closest
    n_closest = predict_n_closest(df, input_ingredients, ingredients, 5)

    # check the first element
    assert n_closest[0] == (28497, 0.424411057990635)
    # check the last element
    assert n_closest[-1] == (19220, 0.35702679879185106)