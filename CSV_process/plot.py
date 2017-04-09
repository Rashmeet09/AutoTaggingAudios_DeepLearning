#Plot histogram - names of tags against their frequency count in the dataset

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter

path_csv = 'annotations_40tags.csv'
df = pd.read_csv(path_csv,header=0)
print "data-frame shape:",df.shape
idxs = np.random.permutation(len(df))
print df.shape[0]
tags = df.columns.values.tolist()

    
column = 4
dict = {}
while (column<=43):     #change according to 188 -> 131 -> 40 tags
    sum =0
    for row_idx, row in enumerate(df.iloc[idxs].itertuples()):
	    sum = sum + row[column]
    print tags[column-1],"-",sum
    dict[tags[column-1]]=sum
    column = column + 1

print dict
x = np.arange(len(dict))
dict = OrderedDict(sorted(dict.items(),key = itemgetter(1),reverse=True))
plt.bar(x,dict.values(),width=1,color='r')
plt.xticks(x,dict.keys())
plt.xlabel('Tags',fontsize=15)
plt.ylabel('Count',fontsize=15)
locs, labels = plt.xticks()
plt.setp(labels,rotation=60,fontsize=12)
ymax = max(dict.values()) + 1
plt.ylim(0,ymax)
plt.show()


'''
Top 40 tags with their frequency count:
{'soft': 1052, 'classical': 4358, 'dance': 649, 'heavy metal': 583, 'pop': 995, 'sitar': 926, 'female': 2067, 'ambient': 1956, 'violin': 1907, 'string': 2842, 'slow': 3547, 'fast': 2331, 'new age': 650, 'synthesizer': 1734, 'bass': 337, 'indian': 1402, 'harpsichord': 1123, 'piano': 2056, 'harp': 623, 'solo': 826, 'flute': 1035, 'electric': 2764, 'eastern': 406, 'beat': 2123, 'jazz': 555, 'guitar': 4861, 'techno': 2954, 'no vocal': 2550, 'singer': 1308, 'drum': 2698, 'opera': 1298, 'country': 541, 'quiet': 1168, 'vocal': 2813, 'choir': 830, 'cello': 575, 'rock': 2371, 'weird': 640, 'male': 2169, 'loud': 1086}
'''
