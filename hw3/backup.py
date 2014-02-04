import urllib
import os
import json
import time
import random

#Get rid of the dir_out variable to save in same folder
"""
dir_out = '/'

if not os.path.exists(dir_out):
    print 'Creating directory %s' % dir_out
    os.mkdir(dir_out)
"""


symbols = ['NFLX','MU','BBY','NEM','CLF','EW']
#The above tickers are (in order), the 3 biggest gainers and losers of 2013 in the S&P500 index

for symbol in symbols:
    time.sleep(.1+random.random())

# Remove extraneous references to $25 price point

    """
    data_out = {}
    if 'more than 25' not in data_out:
        data_out['more than 25']=0
    if 'less than 25' not in data_out:
            data_out['less than 25']=0
    """

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

        """
        for d in data:
            d = d.split(',')
            try:
                close = d[6]
                if float(close) >= float(25):
                    data_out['more than 25']+=1
                elif float(close) < float(25):
                    data_out['less than 25']+=1
            except:pass
        """

#        print symbol,':', json.dumps(data_out)
    except:
        print 'failed for %s' % symbol
