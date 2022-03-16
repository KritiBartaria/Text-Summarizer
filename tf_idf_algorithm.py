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


text1 = '''
Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.
I’ve coached mummy and mom clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”
I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.
“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don’t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you’re willing to put yourself on the line or settle for less. And that’s fine if you’re content to receive less, as long as you’re not regretful later.
If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.
Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I’m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you’re an overnight sensation, to sustain it for long, particularly if you don’t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success — simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.
So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what’s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don’t leave your dreams to chance.
'''


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


# In[27]:


'''
app = Flask(__name__)
@app.route('/templates', methods =['POST'])
def original_text_form():
    text = request.form['input_text']
    #number_of_sent = request.form['num_sentences']
    # print("TEXT:\n",text)
    summary = run_summarization(text)
    # print("*"*30)
    # print(summary)
    return render_template('index_k.html', title = "Summarizer", original_text = text, output_summary = summary, num_sentences = 5)

@app.route('/')
def homepage():
    title = "Text Summarizer Tool"
    return render_template('index_k.html', title = title)
    
if __name__ == "__main__":
    app.run()
'''


# In[ ]:





# In[ ]:




