import pytest
from project2 import read_data

def test_read_data():
    df = read_data()
    assert df.head().shape == (5, 3)

def test_read_data_first_row():
    df = read_data()
    assert df.iloc[0]['cuisine'] == 'greek'
    assert df.iloc[0]['id'] == 10259
    assert df.iloc[0]['ingredients'] == ['romaine lettuce', 'black olives', 'grape tomatoes', 'garlic', 'pepper', 'purple onion', 'seasoning', 'garbanzo beans', 'feta cheese crumbles']