#from raw_corpus import *
from goose import Goose
g = Goose()
import inflect
p = inflect.engine()
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import json

def json_to_urls(file):
    urls = []
    json_data = open(file)
    data = json.load(json_data)
    items = data['items']
    for item in items:
        urls.append(item['originId'])
    return urls

def RemoveEndLineDot(sents):
    for sent in sents:
        if sent[-1] == '.':
            sent.pop()
    return sents

def Num2Word(matchObj):
  matchString1 = matchObj.group(0)
  return p.number_to_words(matchString1)

#SUBJECT = 'CNN_Tech'

def CreateCorpus(SUBJECT,UrlList=None):

    OrgCorpusFile = open('OriginalCorpus_'+SUBJECT+'.txt','w')
    ProcessedFile = open('ProcessedFile_'+SUBJECT+'.txt','w')
    file = SUBJECT+'.json'
    if UrlList == None:
        urls = json_to_urls(file)
    else:
        urls = UrlList

    #corpus= []
    count = 0

    for url in urls:
        article = g.extract(url=url)
        clean_text = article.cleaned_text
        decoded_str = clean_text.encode('ascii', 'ignore')

        splited_text = decoded_str.split('\n\n')

        for paragraph in splited_text:
            if paragraph == '' or paragraph.find('http')>=0:
                continue
            OrgCorpusFile.writelines(paragraph+'\n')

            a = re.sub(r'\([^)]*\)', '', paragraph)         # remove content within parenthesis ()
            a = re.sub(r'&',r'and',a)
            a = re.sub(r'\+',r' plus ',a)
            a = re.sub(r'(\.)(com|biz|org|net)',r' dot \2',a)
            a = re.sub(r'(\.)(co)(\.)(uk)',r' dot \2 dot \4',a)
            a = re.sub(r'([0-9]+):([0-9]+)',r'\1 \2',a)     # transform 4:1 to 4 1

            a = re.sub(r'([0-9]+)([a-zA-Z]+)',r'\1 \2',a)   # transform 54B to 54 B
            a = re.sub(r'([a-zA-Z]+)([0-9]+)',r'\1 \2',a)   # transform A05 to A 05

            a = re.sub(r';|:|[!]+|[\.]+','.',a)                        # subtitute ; : to .
            a = re.sub(r',|--|"|#|<|>|~|\||`|\]|\[','',a)                     # remove all this characters (,)(--,",#,<,>,~,|,`)
            a = re.sub(r'-|\*|/|@',' ', a)

            a = re.sub(r'(\$)([0-9]+)\.([0-9]+)\s(million|billion)',r'\2 point \3 \4 dollar',a) # '$1.3 million' -> '1 point 3 million dollar'
            a = re.sub(r'(\$)([0-9]+) (million|billion)',r'\2 \3 dollar',a) # '$1 million' -> '1 million dollar'

            a = re.sub(r'(\$)([0-9]+)',r'\2 dollar',a)

            a = re.sub('\s+',' ',a)
            if a[0] == ' ' or a[0] == '\'':
                a = a[1:]

            splitted_paragraph = [word_tokenize(t) for t in sent_tokenize(a)]
            splitted_paragraph = RemoveEndLineDot(splitted_paragraph)

            for par in splitted_paragraph:

                for word_id in range(len(par)):
                    if word_id == 0 and par[word_id][0] == '\'' and len(par[word_id])>1:
                        par[word_id] = par[word_id][1:]
                    if par[word_id] == '?' or par[word_id] == '\'':
                        par[word_id] = u''
                    if (par[word_id] == "'s" or par[word_id] == "'re" or par[word_id] == "'ll" or par[word_id] == "'" or
                            par[word_id] == "'ve" or par[word_id] == "'m" or par[word_id] == "'d") and word_id !=0:

                        par[word_id-1] = par[word_id-1] + par[word_id]
                        par[word_id] = u''


                par = [p.number_to_words(w) if w.isdigit() else w.lower() for w in par]
                par = " ".join(par)
                par = re.sub(r'([0-9]+)(st|nd|rd|th)',Num2Word,par)
                par = re.sub(r'([0-9]+).([0-9]+)',Num2Word,par)
                par = re.sub(r'\s+',' ',par)
                par = re.sub(r'-',' ', par)
                par = re.sub(r' n\'t','n\'t', par)
                par = re.sub(r'%','percents',par)

                #corpus.append(par) # write to file ?
                ProcessedFile.writelines(par+'\n')

        OrgCorpusFile.writelines('\n')
        ProcessedFile.writelines('\n')
        count += 1
        print "Article Number", count, "out of", len(urls)
    print "Finished."

    OrgCorpusFile.close()
    ProcessedFile.close()