import csv
import re
from collections import defaultdict
from string import printable
from spell_check import correct
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
import math
import random
from operator import itemgetter
f = open('w3_dict.json')
l = f.readline()
f.close()
word_freq_3 = eval(l)

f = open('w4_dict.json')
l = f.readline()
f.close()
word_freq_4 = eval(l)

f = open('w5_dict.json')
l = f.readline()
f.close()
word_freq_5 = eval(l)
#def extract_dictionary(file):
#    for item in open(file):
#        item = item.rstrip('\n').split()
#        freq = int(item[0])
#        elem = '|'.join(item[1:])
#        word_freq[elem] = freq


def calculate_frequency(trigrams, frequency):
    average_frequency = 0.0
    count = 0
    percentage = 0.0
    for elem in trigrams:
        if elem in frequency:
            average_frequency = average_frequency + math.log(frequency[elem])
            count = count + 1
    if len(trigrams)>0:
        average_frequency = average_frequency/(len(trigrams) * 1.0)
        percentage = count / (len(trigrams) * 1.0)
    else:
        average_frequency = 0.0
        percentage = 0.0
    
    return average_frequency, percentage

def ngrams(text, n):
    text = text.split()
    return ['|'.join(text[i:i + n]) for i in range(0, len(text) - n + 1)]
    
#
##NOTE: Considering only essay set 1 for now
#def main():
main_unary_text_dict=defaultdict(lambda: 0)
main_unary_postag_dict=defaultdict(lambda: 0)
main_unary_stemmed_dict=defaultdict(lambda: 0)
idf_unary_text_dict = defaultdict(lambda: 0)
gold_student = []
gold_2_student = []
gold_1_student = []
gold_0_student = []
documents = 0
#    st = LancasterStemmer()
porter_st = PorterStemmer()
data = csv.reader(open('train_test1.tsv'), delimiter='\t')
processed_data = {}
for index, item in enumerate(data):
    # Pre processing phase
    if index == 0:
        continue
    if item[2] == '4':
        id = int(item[0])
        processed_data[id] = {}
        elem = processed_data[id]
        elem['essay_set'] = int(item[1])
        elem['score1'] = int(item[2])
        elem['score2'] = int(item[3])
        elem['raw_text'] = item[4]
        
        #Remove non printable characters (\n \t \r etc)
        #Convert to lowercase
        raw_text = item[4]
        text = [k.lower() for k in raw_text if k in printable and k not in ['\n', '\t', '"']]
        text = ''.join(text)
        #Correct the spelling
        #Remove all non alpha numeric characters (., etc)
        text = re.sub("[^0-9a-z\s]+","",text)
        temp = text.split()
        wrong_spellings = 0
        for x in range(0,len(temp)):
            s = correct(temp[x])
            if s is not temp[x] and not temp[x].isdigit():
               temp[x] = s 
               wrong_spellings += 1
        elem['word_count'] = len(temp)
        text=' '.join(temp)
        elem['wrong_spelling']=wrong_spellings
        elem['cleaned_text'] = text
        #POS tagging
        temp = pos_tag(text.split())
        pos_tagged = []
        verb_count = 0
        conjunction_count = 0
        adjective_count = 0
        noun_count = 0
        length_4_more = 0
        length_6_more = 0
        length_8_more = 0 
        for pos in temp:
            if len(pos[0]) >= 4 and  len(pos[0]) < 6:
                length_4_more += 1
            elif len(pos[0]) >= 6 and  len(pos[0]) < 8:
                length_6_more += 1
            elif len(pos[0]) >= 8:
                length_8_more += 1
            pos_tagged.append(pos[1])
            if pos[1] in ['NN', 'NNP', 'NNPS', 'NNS']:
                noun_count += 1
            elif pos[1] in ['CC']:
                conjunction_count += 1
            elif pos[1] in ['VB', 'VBD','VBG','VBN','VBP','VBZ']:
                verb_count += 1
            elif pos[1] in ['JJ', 'JJR', 'JJS']:
                adjective_count += 1
        
        elem['length_4_more'] = length_4_more
        elem['length_6_more'] = length_6_more
        elem['length_8_more'] = length_8_more
        elem['noun_count'] = noun_count
        elem['conjunction_count'] = conjunction_count
        elem['verb_count'] = verb_count
        elem['adjective_count'] = adjective_count
        pos_tagged = ' '.join(pos_tagged)
        elem['pos_tagged'] = pos_tagged
        unary_postag_dict=defaultdict(lambda: 0)
        for words in pos_tagged.split():
            if words in ['.', '', ' ']:
               continue
            unary_postag_dict[words]=unary_postag_dict[words]+1
        elem['unary_postag_dict']=unary_postag_dict
        ###################################################################
        text = re.sub("[^a-zA-Z\s]"," ",text) # Removal of non alphabetic characters
        elem['text_wo_numbers'] = text
        ###################################################################
        #Removal of stop words
        stop_words = stopwords.words('english')
        filtered_words = [k for k in text.split() if k not in stop_words]
        filtered_words = ' '.join(filtered_words)
        elem['text_wo_stop_words'] = filtered_words
        unary_text_dict=defaultdict(lambda: 0)
        for words in filtered_words.split():
            if words in ['.', '', ' ']:
               continue
            unary_text_dict[words]=unary_text_dict[words]+1
        elem['unary_text_dict']=unary_text_dict
#        for key in unary_text_dict:
#            idf_unary_text_dict[key] += 1
        #Stemming of the words
        porter_st_stemmed_text = [porter_st.stem(k) for k in filtered_words.split()]
        porter_st_stemmed_text = ' '.join(porter_st_stemmed_text)
        elem['stemmed_text'] = porter_st_stemmed_text
        #tri grams of stemmed words        
    #   elem['stemmed_text_trigrams'] = ngrams(porter_st_stemmed_text, 3)
        unary_stemmed_dict=defaultdict(lambda: 0)
        for words in porter_st_stemmed_text.split():
            if words in ['.', '', ' ']:
               continue
            unary_stemmed_dict[words]=unary_stemmed_dict[words]+1
        elem['unary_stemmed_dict']=unary_stemmed_dict
        ##################################################################
        pattern = re.compile('[+-]?\d+[\.]?\d*')
        numbers = [item for item in text.split() if pattern.match(item)]
        numbers = ' '.join(numbers)
        elem['numbers'] = numbers
    else:
        documents +=1
        id = int(item[0])
        processed_data[id] = {}
        elem = processed_data[id]
        elem['essay_set'] = int(item[1])
        elem['score1'] = int(item[2])
        elem['score2'] = int(item[3])
        elem['raw_text'] = item[4]
        if elem['score1'] == elem['score2'] and elem['score1'] == 3:
            gold_student.append(id)
            
        if elem['score1'] == elem['score2'] and elem['score1'] == 2:
            gold_2_student.append(id)
        
        if elem['score1'] == elem['score2'] and elem['score1'] == 1:
            gold_1_student.append(id)
            
        if elem['score1'] == elem['score2'] and elem['score1'] == 0:
            gold_0_student.append(id)
        #Remove non printable characters (\n \t \r etc)
        #Convert to lowercase
        raw_text = item[4]
        text = [k.lower() for k in raw_text if k in printable and k not in ['\n', '\t', '"']]
        text = ''.join(text)
        #Correct the spelling
        #Remove all non alpha numeric characters (., etc)
        text = re.sub("[^0-9a-z\s]+","",text)
        temp = text.split()
        wrong_spellings = 0
        for x in range(0,len(temp)):
            s = correct(temp[x])
            if s is not temp[x] and not temp[x].isdigit():
               temp[x] = s 
               wrong_spellings += 1
        elem['word_count'] = len(temp)
        text=' '.join(temp)
        elem['wrong_spelling']=wrong_spellings
        elem['cleaned_text'] = text
        #POS tagging
        temp = pos_tag(text.split())
        pos_tagged = []
        verb_count = 0
        conjunction_count = 0
        adjective_count = 0
        noun_count = 0
        length_4_more = 0
        length_6_more = 0
        length_8_more = 0 
        for pos in temp:
            if len(pos[0]) >= 4 and  len(pos[0]) < 6:
                length_4_more += 1
            elif len(pos[0]) >= 6 and  len(pos[0]) < 8:
                length_6_more += 1
            elif len(pos[0]) >= 8:
                length_8_more += 1
            pos_tagged.append(pos[1])
            if pos[1] in ['NN', 'NNP', 'NNPS', 'NNS']:
                noun_count += 1
            elif pos[1] in ['CC']:
                conjunction_count += 1
            elif pos[1] in ['VB', 'VBD','VBG','VBN','VBP','VBZ']:
                verb_count += 1
            elif pos[1] in ['JJ', 'JJR', 'JJS']:
                adjective_count += 1
        
        elem['length_4_more'] = length_4_more
        elem['length_6_more'] = length_6_more
        elem['length_8_more'] = length_8_more
        elem['noun_count'] = noun_count
        elem['conjunction_count'] = conjunction_count
        elem['verb_count'] = verb_count
        elem['adjective_count'] = adjective_count
        pos_tagged = ' '.join(pos_tagged)
        elem['pos_tagged'] = pos_tagged
        unary_postag_dict=defaultdict(lambda: 0)
        for words in pos_tagged.split():
            if words in ['.', '', ' ']:
               continue
            main_unary_postag_dict[words] = main_unary_postag_dict[words] + 1
            unary_postag_dict[words]=unary_postag_dict[words]+1
        elem['unary_postag_dict']=unary_postag_dict
        ###################################################################
        text = re.sub("[^a-zA-Z\s]"," ",text) # Removal of non alphabetic characters
        elem['text_wo_numbers'] = text
        ###################################################################
        #Removal of stop words
        stop_words = stopwords.words('english')
        filtered_words = [k for k in text.split() if k not in stop_words]
        filtered_words = ' '.join(filtered_words)
        elem['text_wo_stop_words'] = filtered_words
        unary_text_dict=defaultdict(lambda: 0)
        for words in filtered_words.split():
            if words in ['.', '', ' ']:
               continue
            main_unary_text_dict[words] = main_unary_text_dict[words] + 1
            unary_text_dict[words]=unary_text_dict[words]+1
        elem['unary_text_dict']=unary_text_dict
        for key in unary_text_dict:
            idf_unary_text_dict[key] += 1
        #Stemming of the words
        porter_st_stemmed_text = [porter_st.stem(k) for k in filtered_words.split()]
        porter_st_stemmed_text = ' '.join(porter_st_stemmed_text)
        elem['stemmed_text'] = porter_st_stemmed_text
        #tri grams of stemmed words        
    #   elem['stemmed_text_trigrams'] = ngrams(porter_st_stemmed_text, 3)
        unary_stemmed_dict=defaultdict(lambda: 0)
        for words in porter_st_stemmed_text.split():
            if words in ['.', '', ' ']:
               continue
            main_unary_stemmed_dict[words] = main_unary_stemmed_dict[words] + 1
            unary_stemmed_dict[words]=unary_stemmed_dict[words]+1
        elem['unary_stemmed_dict']=unary_stemmed_dict
        ##################################################################
        pattern = re.compile('[+-]?\d+[\.]?\d*')
        numbers = [item for item in text.split() if pattern.match(item)]
        numbers = ' '.join(numbers)
        elem['numbers'] = numbers
    
    if index%100 == 0:
        print(index)

#Finding unigrams features of raw_text, stemmed_text, pos tags

f = open("main_unary_text_dict_1", "w")
f.write(str(dict(main_unary_text_dict)))
f.close()
f = open("main_unary_postag_dict_1", "w")
f.write(str(dict(main_unary_postag_dict)))
f.close()
f = open("main_unary_stemmed_dict_1", "w")
f.write(str(dict(main_unary_stemmed_dict)))
f.close()

main_unary_text_dict = dict(sorted(dict(main_unary_text_dict).items(), key=itemgetter(1), reverse=True)[:10])
main_unary_postag_dict = dict(sorted(dict(main_unary_postag_dict).items(), key=itemgetter(1))[10:20])
main_unary_stemmed_dict = dict(sorted(dict(main_unary_stemmed_dict).items(), key=itemgetter(1), reverse=True)[:10])


csvDump = 'score1,wrong_spellings,word_count,adjective_count,conjunction_count,noun_count,verb_count,length_4_more,length_6_more,length_8_more,'
main_text_keys= []
main_postag_keys = []
main_stem_keys = []
for key in main_unary_text_dict:
    csvDump += key +','
    main_text_keys.append(key)
for key in main_unary_postag_dict:
    csvDump += key +','
    main_postag_keys.append(key)
for key in main_unary_stemmed_dict:
    csvDump += key +','
    main_stem_keys.append(key)
    
    





for each in idf_unary_text_dict:
    idf_unary_text_dict[each] = 1 +  math.log(documents/idf_unary_text_dict[each])
    
   

#gold Standard students
random.shuffle(gold_student)
random.shuffle(gold_2_student)
random.shuffle(gold_1_student)
random.shuffle(gold_0_student)
gold_student = gold_student[0:10]
gold_2_student = gold_2_student[0:10]
gold_1_student = gold_1_student[0:10]
gold_0_student = gold_0_student[0:10]
gold_tfidf = {}

keyOrder = []
for i in gold_student:
    total = 0
    keyOrder.append(i)
    csvDump += 'gold_student_' + str(i) +','
    for each in processed_data[i]['unary_text_dict']:
        total += processed_data[i]['unary_text_dict'][each]
    gold_tfidf[i] = {}
    for each in processed_data[i]['unary_text_dict']:
        gold_tfidf[i][each] = processed_data[i]['unary_text_dict'][each]/total * idf_unary_text_dict[each]

for i in gold_2_student:
    total = 0
    keyOrder.append(i)
    csvDump += 'gold_2_student_' + str(i) +','
    for each in processed_data[i]['unary_text_dict']:
        total += processed_data[i]['unary_text_dict'][each]
    gold_tfidf[i] = {}
    for each in processed_data[i]['unary_text_dict']:
        gold_tfidf[i][each] = processed_data[i]['unary_text_dict'][each]/total * idf_unary_text_dict[each]

for i in gold_1_student:
    total = 0
    keyOrder.append(i)
    csvDump += 'gold_1_student_' + str(i) +','
    for each in processed_data[i]['unary_text_dict']:
        total += processed_data[i]['unary_text_dict'][each]
    gold_tfidf[i] = {}
    for each in processed_data[i]['unary_text_dict']:
        gold_tfidf[i][each] = processed_data[i]['unary_text_dict'][each]/total * idf_unary_text_dict[each]

for i in gold_0_student:
    total = 0
    keyOrder.append(i)
    csvDump += 'gold_0_student_' + str(i) +','
    for each in processed_data[i]['unary_text_dict']:
        total += processed_data[i]['unary_text_dict'][each]
    gold_tfidf[i] = {}
    for each in processed_data[i]['unary_text_dict']:
        gold_tfidf[i][each] = processed_data[i]['unary_text_dict'][each]/total * idf_unary_text_dict[each]


csvDump += "3gram_freq,3gram_percentage,4gram_freq,4gram_percentage,5gram_freq,5gram_percentage"
csvDump = csvDump.strip(',')
csvDump += '\n'   
        

for i in sorted(processed_data):
    field_name  = 'text_wo_stop_words_feature'
    u='unary_text_dict'  
    main_k=''
    csvDump += str(processed_data[i]['score1']) + ',' + str(processed_data[i]['wrong_spelling']) +',' + str(processed_data[i]['word_count']) + ','   
    csvDump +=  str(processed_data[i]['adjective_count']) + ',' + str(processed_data[i]['conjunction_count']) +',' + str(processed_data[i]['noun_count']) + ',' + str(processed_data[i]['verb_count']) + ','
    csvDump +=  str(processed_data[i]['length_4_more']) + ',' + str(processed_data[i]['length_6_more']) +',' + str(processed_data[i]['length_8_more']) + ',' 
    for each in main_text_keys:
        if each in processed_data[i][u]:
           main_k=main_k+str(processed_data[i][u][each])+','
          
        else:
           main_k=main_k+'0,'
    csvDump += main_k 
  
    main_k = main_k.strip(',')
    processed_data[i][field_name] = main_k
    
    #tf for student i    
    total =0
    for each in processed_data[i][u]:
        total += processed_data[i][u][each]
    tfidf = defaultdict(lambda: 0)
    for each in processed_data[i][u]:
        if total>0 and idf_unary_text_dict[each] >0:
            tfidf[each] = processed_data[i][u][each]/total * idf_unary_text_dict[each]
        else:
            tfidf[each] = 0
    #Cosine simlarity with each gold_standard
    cosine_similarity = ''
    for j in keyOrder:
        dot_product = 0
        query = 0
        document = 0
        for each in gold_tfidf[j]:
           dot_product += gold_tfidf[j][each]*tfidf[each]
           document += math.pow(gold_tfidf[j][each],2)
           query += math.pow(tfidf[each],2)
        if query==0.0:
            similarity =0.0
        else:
            similarity = dot_product/(math.sqrt(document)*math.sqrt(query))
        cosine_similarity += str(similarity) + ','
    
    field_name  = 'pos_tagged_feature'
    u='unary_postag_dict'  
    main_k=''
    for each in main_postag_keys:
        if each in processed_data[i][u]:
           main_k=main_k+str(processed_data[i][u][each])+','
        else:
           main_k=main_k+'0,'
    csvDump += main_k 
    main_k = main_k.strip(',')
    processed_data[i][field_name] = main_k
    field_name  = 'stemmed_text_feature'
    u='unary_stemmed_dict'  
    main_k=''
    for each in main_stem_keys:
        if each in processed_data[i][u]:
           main_k=main_k+str(processed_data[i][u][each])+','
        else:
           main_k=main_k+'0,'
    csvDump += main_k 
    csvDump += cosine_similarity    
    for c in [3,4,5]:
        text = processed_data[i]['cleaned_text']
        tri_grams = ngrams(text, c)
        if c == 3:
            [frequency, percentage] = calculate_frequency(tri_grams, word_freq_3)
        elif c == 4:
            [frequency, percentage] = calculate_frequency(tri_grams, word_freq_4)
        else:
            [frequency, percentage] = calculate_frequency(tri_grams, word_freq_5)
        csvDump += str(frequency) + ',' + str(percentage) + ','
    main_k = main_k.strip(',')
    processed_data[i][field_name] = main_k 
    csvDump = csvDump.strip(',') + '\n'

f = open('csvDump1.csv', 'w')
f.write(csvDump)
f.close()

f = open('processedData1.dict','w')
f.write(str(processed_data))
f.close()    

#main()


