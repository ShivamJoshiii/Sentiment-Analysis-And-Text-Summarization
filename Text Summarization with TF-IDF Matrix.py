import json
data = [json.loads(line) for line in open("E:\MS SEM2\CIS 593\Project\Dataset\yelp_academic_dataset_review.json", "r", encoding="utf8", errors="ignore")]

import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

#This function finds the count of each word and creates frequency matrix for each word
def frequency_table(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1
        frequency_matrix[sent[:15]] = freq_table
    return frequency_matrix

#This fucntion gets the word fequency matrix and make TF matrix 
def get_tf_matrix(freq_matrix):
    tf_matrix = {}
    for sent, f_table in freq_matrix.items():
        tf_table = {}
        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence
        tf_matrix[sent] = tf_table
    return tf_matrix

#This function creates word per document table which will be used for IDF matrix creation
def documents_per_words_count(freq_matrix):
    word_per_doc_table = {}
    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
    return word_per_doc_table

#This function uses the word per document table and gets other two parameters and finds IDF matrix
def get_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}
    for sent, f_table in freq_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))
        idf_matrix[sent] = idf_table
    return idf_matrix

#This is simple multiplicaiton of TF matrix and IDF matrix to find TF-IDF scores for words
def tf_idf_matrix_multiplication(tf_matrix, idf_matrix):
    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table
    return tf_idf_matrix

#Based on the TF-IDF score, each sentence will be valued
def sentence_score(tf_idf_matrix) -> dict:
    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
    return sentenceValue

#Find the average score - define as threashold
def _find_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))
    return average

#This function gets each sentence, its value and threashold and generate the summary
def summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary

#This fucntion calls each supporting function to generate summary
def run_summarization(text):
    sentences = sent_tokenize(text)
    total_documents = len(sentences)
    freq_matrix = frequency_table(sentences)
    tf_matrix = get_tf_matrix(freq_matrix)
    count_doc_per_words = documents_per_words_count(freq_matrix)
    idf_matrix = get_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    tf_idf_matrix = tf_idf_matrix_multiplication(tf_matrix, idf_matrix)
    sentence_scores = sentence_score(tf_idf_matrix)
    threshold = _find_average_score(sentence_scores)
    summaryValue = summary(sentences, sentence_scores, 1.0 * threshold)
    return summaryValue

#If you wanna get the summaried text, run me
if __name__ == '__main__':
    result = run_summarization(data[5502]['text'])
    print(result)
    