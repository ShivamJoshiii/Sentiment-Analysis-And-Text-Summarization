import json
data = [json.loads(line) for line in open("E:\MS SEM2\CIS 593\Project\Dataset\yelp_academic_dataset_review.json", "r", encoding="utf8", errors="ignore")]

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

#This function reads the whole article and splite it into the sentenses.
def read_article(file):
    article = file.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    return sentences

#This function finds the similarity between all the combination of sentenses
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
        
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
#This fucntion builds similarity matrix based on the above funtion - sentence_similarity    
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

#this is the main function which calls all other required funtion to generate summary
def generate_summary(file_name, top_n):
    stop_words = stopwords.words('english')
    summarize_text = []

    sentences =  read_article(file_name)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    print("Summarize Text: \n", ". ".join(summarize_text))

# Call the functoin here to generate the summary
    #If you wanna get the summaried text, run me
if __name__ == '__main__':
    generate_summary( data[5502]['text'], 6)