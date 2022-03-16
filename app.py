#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tf_idf_algorithm as t
from werkzeug.wrappers import Request, Response
from flask import Flask, request, render_template
from werkzeug.serving import run_simple


# In[7]:


app = Flask(__name__)
@app.route('/templates', methods =['POST'])
def original_text_form():
    text = request.form['input_text']
    #number_of_sent = request.form['num_sentences']
    # print("TEXT:\n",text)
    summary = t.run_summarization(text)
    # print("*"*30)
    # print(summary)
    return render_template('index_k.html', title = "Summarizer", original_text = text, output_summary = summary, num_sentences = 5)

@app.route('/')
def homepage():
    title = "Text Summarizer Tool"
    return render_template('index_k.html', title = title)


# In[ ]:


if __name__ == "__main__":
    app.run()

