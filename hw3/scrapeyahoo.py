import urllib
import os
import json
import time
import random
import numpy as np

symbols = ['BTH','ZUMZ','NFLX','GTIV','BBY']
#The above tickers are (in order), the 5 most volatile stocks of 2013

for symbol in symbols:
    time.sleep(.1+random.random())

    try:
        url = 'http://ichart.finance.yahoo.com/table.csv?s=%s&a=00&b=1&c=2013&d=11&e=31&f=2013&g=d&ignore=.csv' % symbol
        print url
        data = urllib.urlopen(url).read()
        file = '%s_data.csv' % (symbol)
        f = open(file,'w')
        f.write('%s' % str(data))
        f.close()
        print 'wrote file %s' % file
        data = data.split('\n')

    except:
        print 'failed for %s' % symbol


