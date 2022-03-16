#!/usr/bin/env python
# coding: utf-8

# In[14]:


import math
from datetime import datetime
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from werkzeug.wrappers import Request, Response
from flask import Flask, request, render_template
from werkzeug.serving import run_simple


# In[15]:



# In[16]:


print('Total number of words in the text are     :', len(text1))
print('Total number of sentences in the text are :', len(text1.split('.')))


# In[17]:


def create_freq_matrix(sentences):
    '''
    todo   : calculate the frequency of words in each sentence.
    input  : array of sentences
    output : frequence matrix
    '''
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
    #Using hash value of the sentences to save memory & have symmetry in the key's of the freq matrix.
    #Each hash value (of the sentence) is the key and the value is a dictionary of word frequency.
        frequency_matrix[hash(sent)] = freq_table
    return frequency_matrix


# In[18]:


def create_tf_matrix(freq_matrix):
    '''
    todo   : calculate the Term-Frequency of each word in a paragraph.
    input  : frequence matrix
    output : term-frequency matrix
    '''
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


# In[19]:


def create_words_per_sentences(freq_matrix):
    '''
    todo   : calculate how many sentences contain a word.
    input  : frequence matrix
    output : words per sentence matrix
    '''
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table


# In[20]:


def create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    '''
    todo   : find the IDF for each word in a paragraph.
    input  : frequence matrix, words per sentence matrix, total number of documents
    output : words per sentence matrix
    '''
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix


# In[21]:


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    '''
    todo   : multiply the values from both the  TF & IDF matrix and generate a new matrix.
    input  : term-frequence matrix, inverse term-frequency matrix
    output : TF-IDF matrix
    '''
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


# In[22]:


def score_sentences(tf_idf_matrix) -> dict:
    '''
    todo   : score a sentence by its word's TF.
    input  : TF-IDF matrix
    output : Scored sentences dictionary.
    '''

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


# In[23]:


def find_average_score(sentenceValue) -> int:
    '''
    todo   : Find the average score from the sentence value dictionary
    input  : Scored sentences dictionary
    output : Threshold value
    '''
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average


# In[24]:


def generate_summary(sentences, sentenceValue, threshold):
    '''
    todo   : generate the summary using threshold value as a floor limit.
    input  : Sentences array, Scored sentences dictionary & threshold
    output : Summary.
    '''
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if hash(sentence) in sentenceValue and sentenceValue[hash(sentence)] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


# In[25]:


def generate_summary_top_n(sentences, sentence_scores, top_n):
    '''
    todo   : generate the summary using top_n sentences based on the decreasing order of their scores.
    input  : Sentences array, Sentence value dictionary & 'n'
    output : Summary.
    '''
    sentence_count = 0
    summary = ''
    sort_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    #print(sort_sentences)
    #print(top_n)
        #summarize_text.append(" ".join(ranked_sentence[i][1]))
    for sentence in sentences:
        if hash(sentence) in sentence_scores and sentence_count <= top_n :
            summary += " " + sentence
            sentence_count += 1

    return summary


# In[26]:


def run_summarization(text):
    '''
    todo   : Execute the above defined functiona in the sequentional order.
    input  : text
    output : Scored sentences dictionary, array of sentences.
    '''

    #to run the sent_tokenize() method to create the array of sentences.
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    #print(len(sentences))

    #Create the Frequency matrix of the words in each sentence.
    freq_matrix = create_freq_matrix(sentences)
    #print(freq_matrix)
   
    #calculate TermFrequency and generate a matrix
    tf_matrix = create_tf_matrix(freq_matrix)
    #print(tf_matrix)
    
    #creating table for documents per words
    count_words_per_sent = create_words_per_sentences(freq_matrix)
    #print(count_words_per_sent)
    
    # calculate IDF and generate a matrix
    idf_matrix = create_idf_matrix(freq_matrix, count_words_per_sent, total_sentences)
    #print(idf_matrix)
  
    # calculate TF-IDF and generate a matrix
    tf_idf_matrix = create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)
   
    # score the sentences
    sentence_scores = score_sentences(tf_idf_matrix)
    #print(sentence_scores)
    
    # Find the threshold
    threshold = find_average_score(sentence_scores)
    #print(threshold)

    # Important Algorithm: Generate the summary
    summary = generate_summary(sentences, sentence_scores, 1.3 * threshold)
    return summary




