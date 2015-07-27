import urllib2
from CNN import CreateCorpus

urls = ['http://news.blogs.cnn.com/tag/cnns-laura-smith-spark/',
        'http://news.blogs.cnn.com/tag/cnns-laura-smith-spark/page/2/',
        'http://news.blogs.cnn.com/tag/cnns-laura-smith-spark/page/3/']

new_urls = []

for url in urls:
    html = urllib2.urlopen(url).read()

    while html.find('<a class="cnn_full_story_link" href="') > 0 :
        start_pointer = html.find('<a class="cnn_full_story_link" href="')
        end_pointer = html.find('" >FULL STORY</a>	')
        new_urls.append(html[start_pointer + len('<a class="cnn_full_story_link" href="'):end_pointer])
        html = html[end_pointer + len('" >FULL STORY</a>	'):]

CreateCorpus('Laura1',new_urls)
###################################################################
import pickle
file = open('ReportersData.pickle', 'r')
ReportersDict,MapReportes2Urls = pickle.load(file)
file.close()

laura_keys = ['Laura Smith-Spark and Sweelin Ong','Masoud Popalzai and Laura Smith-Spark',
  'Michael Schwartz and Laura Smith-Spark','Elwyn Lopez and Laura Smith-Spark',
   'Laura Smith-Spark and Radina Gigova','Nicola Goulding and Laura Smith-Spark','Laura Smith-Spark and Hada Messia',
     'Barbie Latza Nadeau and Laura Smith-Spark','Greg Botelho and Laura Smith-Spark','Laura Smith-Spark']

urls = []
for keys in laura_keys:
    urls.extend(MapReportes2Urls[keys])

CreateCorpus('Laura2',urls)
###################################################################
