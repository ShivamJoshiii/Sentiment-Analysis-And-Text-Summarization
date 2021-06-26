import json
data = [json.loads(line) for line in open("E:\MS SEM2\CIS 593\Project\Dataset\yelp_academic_dataset_review.json", "r", encoding="utf8", errors="ignore")]

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

#This function gets the text and word frequency table
def frequency_table(text_string) -> dict:
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    return freqTable

#This fucntion score the sentence based on teh frequency of the word in it
def sentences_score(sentences, freqTable) -> dict:
    sentenceValue = dict()
    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]
        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence
    return sentenceValue

#Get the average score - threshold
def get_average_score(sentenceValue) -> int:
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))
    return average

#This function finds the summary and return to the run_summarization funciton
def summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''
    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1
    return summary

#this function does all the steps of Text summerization by callin each funciton saperately
def run_summarization(text):
    freq_table = frequency_table(text)
    sentences = sent_tokenize(text)
    sentence_scores = sentences_score(sentences, freq_table)
    threshold = get_average_score(sentence_scores)
    summaryText = summary(sentences, sentence_scores, 1.5 * threshold)
    print(summaryText)

#if you wan't summary, run me
if __name__ == '__main__':
    run_summarization(data[5502]['text'])
