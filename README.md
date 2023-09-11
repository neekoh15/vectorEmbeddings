# Semantic Search with Vector Embeddings
This repository provides a high-level overview of the application designed to work with text embeddings and search for matches in a "Questions and Answers" database.

## VModel Module
The VModel module is crafted to handle text embeddings and search for matches within a "Questions and Answers" dataset.

## Initialization
Loads the JSON file containing the questions and answers.
Initializes the tokenizer and the ALL-MINI-L6-V2 model using the transformers library.
Processes the data and prepares the FAISS index for rapid searches.
## Data Preprocessing
Loads data from the JSON file.
Extracts all questions and prepares the context for subsequent vectorization.
## Model Loading
Vectorizes the questions database.
Adds a FAISS index to the dataset to enable rapid and efficient searches within the embeddings space.
## Mean Pooling
A technique to consolidate word embeddings into a sentence/document embedding. It averages the embeddings of the words, weighted by their attention, to obtain a single vector representing the entire text.

## Embeddings Retrieval
Tokenizes the text and fetches its vector representation using the model.

## Best Matches Search
Searches the dataset for questions that best match the user's query, based on the similarity of the embeddings.

## JSON Query
The main function that takes a user's query, searches for the most similar question in the dataset, and returns the corresponding answer.

The module concludes with a loop allowing users to make queries and receive real-time answers.
