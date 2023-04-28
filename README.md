# CS5293 Spring 2023 Project 2

# Description
Ever wondered if you can just look into your fridge and kitchen closet to see what ingredients you have, and depending on it you can create a recipe? Well, now you can. This project is a recipe generator that takes in a list of ingredients and outputs a cuisine id similar to those ingredients from the yummly.json. 

This project is built on a trained model that takes a bunch of ingredients and outputs its similarity score to a cuisine. The model is trained on the yummly.json dataset. The model is trained using the KNN model.

Below is the tree for the project:
├── COLLABORATORS
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── docs
│   └── yummly.json
├── model.json
├── project2.py
├── setup.cfg
├── setup.py
└── tests
    ├── test_project2.py
    └── test_read_data.py

# How to install
To install grab the github link and git clone it.

# How to run
The project has two types of inputs, ```--ingredient```-->The ingredients you want to build a cuisine upon and ```--N```-->The amount of similar cuisines you want according to the ingredients given. An example command is ```pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies"```. In order to run the tests you can use the command ```pipenv run python -m pytest```.

# External Libraries
The external libraries used are as follows:
```sklearn```: sklearn or sci-kit is a python library that allows you to train models with Machine learning and vectorization. We use this library to train our model.
```numpy```: numpy is a python library that allows you to do vectorization and matrix operations. We use this library to vectorize our data.
```pandas```: pandas is a python library that allows you to read data from a csv file and convert it into a dataframe. We use this library to read the yummly.json file.
```json```: json is a python library that allows you to write json format text. We use this to output the json format
```pytest```: pytest is a python library that allows you to run tests. We use this library to run our tests.
```argparse```: Used to read inputs from the command line

# Demo
The demo is in the docs folder as a gif. Due to size limitations it is very short. 
![alt-text](https://github.com/fazilraja/cs5293sp23-project2/blob/main/docs/project2demo.gif)
# Functions
The functions are as follows:

## read_data
This function reads the yummly.json file and converts it into a dataframe. It then returns the dataframe.

## prints_json
Given the cuisine, its score and closest N cuisines, it prints it in the json format given such as
```
{
  "cuisine": "America",
  "score": 0.91,
  "closest": [
    {
      "id": "10232",
      "score": 0.34
    },
    {
      "id": "10422",
      "score": 0.15
    },
    {
      "id": "45",
      "score": 0.13
    },
    {
      "id": "7372",
      "score": 0.04
    },
    {
      "id": "9898",
      "score": 0.02
    }
  ]
}```

## get_corpus
This function takes in the dataframe and returns the corpus of the ingredients. We designed the corpus to have all of the ingredients in yummly.json. The last element in the corpus are the ingredients in the input. This is done so that we can vectorize the data and train the model.

## predict_cuisine
This function predicts the type of cuisine from the ingredients. It takes in the dataframe, corpus and the number of cuisines you want to output. It then vectorizes the data and trains the KNN model. It then returns the cuisine, its score and the closest N cuisines.

## predict_n_closest
Given the ingredients and every cuisines ingredients in the yummly.json, it returns the closest N cuisines using cosine similarity.

## main
This function is the main function that takes in the inputs from the command line and runs the functions above.

# Tests
We have 2 test files, mainly for the fact that due to the nature of the project, we have to test the functions that read the data and the functions that predict the cuisine. Designing the tests were difficult due to the nature of training the model and predicting the outputs. The tests are as follows:

## test_read_data file
This test tests the read_data function. It tests if the function returns a dataframe and if the dataframe has the correct columns

## test_project2 file

### test_get_input
This test tests if the function returns the correct inputs from the command line

### test_get_corpus
This test tests if the function returns the correct corpus given the inputs

### test_predict_cuisine
This test tests if the function returns the correct cuisine its score

### test_predict_n_closest
This test tests if the function returns the correct closest N cuisines

# Bugs and Assumptions
One bug would be if the ingredient are not in the training model then it would give an incorrect output. 

Another would be if an ingredient is misspelled then it would give an incorrect output. One assumption is that the ingredients are spelled correctly and are in the training model.

When first run it takes a while to train the model and predict the output. This is due to the fact that the model is trained on a large dataset, so it is assumed that the user is patient when training the model, the program is not broken.