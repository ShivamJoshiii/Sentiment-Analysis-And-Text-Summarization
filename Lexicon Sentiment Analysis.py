import json 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

data = [json.loads(line) for line in open("E:\MS SEM2\CIS 593\Project\Dataset\yelp_academic_dataset_review.json", "r", encoding="utf8", errors="ignore")] 

true = 0
false = 0
total = 0

full_negative = 0
full_neutral  = 0
full_positive = 0

positive = []
with open("E:\\MS SEM2\\CIS 593\\Project\\Dataset\\positive_words.txt",'r') as f:
    q = f.read().split()
    for i in q:
        positive.append(i)
        commmon_positive = set(q)


negative = []
with open("E:\\MS SEM2\\CIS 593\\Project\\Dataset\\negative_words.txt",'r') as f:
    q_one = f.read().split()
    for i in q_one:
        negative.append(i)
        commmon_negative = set(q_one)

for i in range(1,10000):
    abcd = []
    abcd = data[i]['text']
    abcd_stars = data[i]['stars']
    abcd = abcd.lower()
    abce = re.sub(r'\d+', '',abcd) # remove numeric value
    abcd = abcd.strip() # removed whitespace
    abcd = re.sub(r'[^\w\s]','',abcd) #removed punctuations
   

    stop_words = set(stopwords.words('english')) # to remove english stop words
    tokens = word_tokenize(abcd) #tokenizing whole file
    result = [i for i in tokens if not i in stop_words]  # for stop words removal
    check_one = set(result)

    positive_test =  commmon_positive & check_one
    negative_test =  commmon_negative & check_one  
    
    a = 0
    for w1 in positive_test:
        word1 = result.count(w1)
        a = a + word1    

    b = 0  
    for w2 in negative_test:
        word2 = result.count(w2)
        b = b + word2
        
    if b > a:
        #print("Review is Negative")
        #print("Real Rating",abcd_stars)
        full_negative = full_negative + 1 
        if(abcd_stars == 1 or abcd_stars == 2):
            true = true + 1
            total = total + 1
        else:
            false =false + 1
            total = total + 1
    elif a == b:
        #print("Review is Neutral")
        #print("Real Rating",abcd_stars)
        full_neutral = full_neutral + 1 
        if(abcd_stars == 3):
            true = true + 1
            total = total + 1
        else:
            false =false + 1
            total = total + 1
    else:
        #print("Review is Positive")
        #print("Real Rating",abcd_stars)
        full_positive = full_positive + 1 
        if(abcd_stars == 4 or abcd_stars == 5):
            true = true + 1
            total = total + 1
        else:
            false =false + 1
            total = total + 1      

print("True Positives- ",true)
print("False Positives- ",false)
print("Total Test Cases- ",total)
print("Accuracy- ",true/total)
