
from django.shortcuts import render
# from scrape import meta_extract
import flask
from flask import Flask, request, render_template, redirect, url_for
# from sentiment import sentiment
import numpy as np
import pandas as pd
import json
import random
import os
import html
import torch
# from linear_politifact_basis import token_basis, convert_word_to_token
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)

from flask import Flask, jsonify, request
from string import punctuation
from collections import Counter
from app import *
from data import data_dict
from meta_article import meta_extract, sentiment


def model_load(net, PATH, name, export=True):
    if export:
        torch.save(net.state_dict(), PATH+name+'.pth')
        return PATH+name+'.pth'
    else:
        net.torch.load_state_dict(torch.load(PATH + name + '.pth'))
        net.eval()

@app.route('/')
def hello():
    return "hello world"

@app.route('/link', methods=["POST", "GET"])
def link():
    if request.method == "POST":
        link_inp = request.form['linker']
        print(type(link_inp))
    
        link_inp = link_inp.replace('.com', 'comkey')
        link_inp = link_inp.replace('https://', 'https')
        link_inp = link_inp.replace('www.', 'www')
        link_inp = link_inp.replace('/', 'slash')
        print(link_inp)
        main = link_inp

        return redirect(url_for("preview_linker", linkage=main))
    else:
        return render_template("link.html")


@app.route(f"/<linkage>")
def preview_linker(linkage):
    preview = linkage
    preview = preview.replace('https', 'https://')
    preview = preview.replace('www', 'www.')
    preview = preview.replace('slash', '/')
    preview = preview.replace('comkey', '.com')

    authart, publ, timg, allimg, tit, summ = meta_extract(preview)
    sent = sentiment(summ)

    inp = tokenize_sequence(summ, token_basis)
    inp = inp[None, :]
    # print(inp.shape)

    feedforward_template = FeedForward(ref_check, inp_size).to(device)
    recurrent_template = RecurrentClassifier(emb_dim, inp_size, 50, ref_check, 2, dropout=0.2).to(device)    
    model_load(feedforward_template, '/Users/devpatelio/Downloads/Coding/Global_Politics_EA/model_parameters/', 'linear_politifact')
    model_load(recurrent_template, '/Users/devpatelio/Downloads/Coding/Global_Politics_EA/model_parameters/', 'lstm_politifact')

    # feedforward_template.eval()
    # recurrent_template.eval()

    output_linear = '0 ERROR'
    output_lstm = '1 ERROR' #check for error without passing error

    output_linear = F.sigmoid(prediction(inp, feedforward_template)).round()
    output_lstm = F.sigmoid(prediction(inp.long(), recurrent_template))

    all_types = list(pd.read_csv(data_dict['politifact_clean'])['veracity'].unique())

    if output_linear == 0:
        output_linear = f"Little Bias: Prediction = {output_linear}"
    elif output_linear == 1:
        output_linear = f"Substantial Bias: Prediction = {output_linear}"

    statement_type = ''
    if output_lstm <= 0.25:
        statement_type = 'True'
    elif 0.25 < output_lstm <= 0.5:
        statement_type = 'Mostly True'
    elif 0.5 < output_lstm <= 0.75:
        statement_type = 'Mostly False'
    elif 0.75 < output_lstm <= 1:
        statement_type = 'False'
    elif output_lstm > 1:
        statement_type = 'Pants on Fire!'

    output_lstm = f"Veracity -> {statement_type}: {output_lstm}"

    # if output_lstm == 0:
    #     output_lstm = f"Limited Veracity: Prediction = {output_lstm}"
    # elif output_lstm == 1:
    #     output_lstm = f"Expressive Veracity: Prediction = {output_lstm}"

    return render_template("preview.html", preview_link=preview,
                                            author_article=authart, 
                                            published_article=publ,
                                            top_image = timg,
                                            all_image = allimg,
                                            title_article=tit,
                                            summary_article=summ,
                                            sentiment=sent,
                                            bias_point=output_linear,
                                            skew_point=output_lstm)



# if __name__ == "__main__":
#     app.run(host='0.0.0.0')

# app.run()
# 


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)