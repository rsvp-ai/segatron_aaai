#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time     :   2020-04-13 21:19
# @Author   :   Richard Bai
# @EMail    :   he.bai@uwaterloo.ca 
import nltk
import os
import json


def sentence_split(line):
        sents = nltk.tokenize.sent_tokenize(line)
        rnt = [sent.split() for sent in sents]
        return rnt


