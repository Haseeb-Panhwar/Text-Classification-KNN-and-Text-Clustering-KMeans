# Text-Classification-KNN-and-Text-Clustering-KMeans

### Running the GUI Application

To run the Streamlit GUI application for evaluating the Vector Space Model (VSM), follow the steps below:

### Navigate to the Application Directory:

1. Open a command prompt or terminal.
2. Change directory to where your app.py file is located.
3. Run the Application:
  i. Type the following command in the terminal: streamlit run app.py
4. Access the Application: Open a web browser and go to the URL provided by Streamlit, typically http://localhost:8501.

Once the application is running, you'll be able to interact with it using the provided input fields and buttons.

### Interacting with the Application

The GUI application allows you to perform the following actions:

### Run KNN: 

Clicking this button executes the K-nearest neighbors (KNN) algorithm for document classification and retrieval. It displays the train and test documents after splitting, as well as the predicted labels and evaluation metrics.

### Run K-means: 

Clicking this button executes the K-means clustering algorithm for document clustering. It shows the random seeds taken, the final clusters, and the evaluation metrics, including the silhouette score graph.

## VSM Model

### Overview

This code implements a basic Vector Space Model (VSM) for text processing and retrieval in Python, incorporating natural language processing (NLP) techniques. The VSM calculates Term Frequency-Inverse Document Frequency (TF-IDF) scores, crucial for ranking and retrieving documents based on their relevance to a given query.

### Features

Text Preprocessing: Cleans text, performs stemming using the PorterStemmer, and removes stopwords.
TF-IDF Calculation: Computes TF-IDF scores for terms in the corpus.
Query Processing: Processes text queries and computes their vector representation based on indexed terms.
Relevance Scoring: Scores and ranks documents using cosine similarity to the query.

### Required Modules

numpy: For handling large arrays and matrices.
json: For storing and retrieving TF-IDF vectors in JSON format.
nltk: Specifically, the PorterStemmer for word stemming.
re: Regular expression operations for text processing.
streamlit: Required for running the vsm_gui.py GUI application.

### File Structure
ResearchPapers/: Directory containing text documents as .txt files, each named with a unique identifier.
doc_tdidf.json : stores td.idf of document in form of vectors in json format. 
Stopword-List.txt: Contains a list of stopwords.
ResearchPapers/: Directory containing text documents as .txt files, each named with a unique identifier.
doc_tdidf.json: Stores TF-IDF vectors of documents in JSON format
