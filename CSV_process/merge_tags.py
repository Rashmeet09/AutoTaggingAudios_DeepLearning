#Merge columns of similar tags in annotations_188tags.csv
#You get 131 tags from original 188 tags after merging. Top 40 have been taken for the experiment.

import csv
import pandas as pd
import numpy as np

df = pd.read_csv('annotations_188tags.csv',header=0)  #the original annotations_final.csv obtained from MTT site renamed

new_column = df['beat'] | df['beats']
df['beat'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['chant'] | df['chanting']
df['chant'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['choir'] | df['choral'] | df['chorus']
df['choir'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['classical'] | df['clasical'] | df['classic']
df['classical'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['drum'] | df['drums']
df['drum'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['electric'] | df['electro'] | df['electronica'] | df['electronic']
df['electric'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['fast'] | df['fast beat'] | df['quick']
df['fast'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['flute'] | df['flutes']
df['flute'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['guitar'] | df['guitars']
df['guitar'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['hard rock'] | df['hard']
df['hard rock'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['harpsichord'] | df['harpsicord']
df['harpsichord'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['heavy metal'] | df['heavy'] | df['metal']
df['heavy metal'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['horn'] | df['horns']
df['horn'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['indian'] | df['india']
df['indian'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['jazz'] | df['jazzy']
df['jazz'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['no beat'] | df['no drums']
df['no beat'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['female'] | df['female singer'] | df['female singing'] | df['female vocals'] | df['female voice'] | df['woman'] | df['women'] | df['woman singing'] | df['female vocal']
df['female'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['male'] | df['male singer'] | df['male vocal'] | df['male vocals'] | df['male voice'] | df['man'] | df['man singing'] | df['men']
df['male'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['no vocal'] | df['no singing'] | df['no singer'] | df['no vocals'] | df['no voice'] | df['no voices'] | df['instrumental']
df['no vocal'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['opera'] | df['operatic']
df['opera'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['orchestral'] | df['orchestra']
df['orchestral'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['quiet'] | df['silence'] | df['calm']
df['quiet'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['singer'] | df['singing']
df['singer'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['spacey'] | df['space']
df['spacey'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['string'] | df['strings']
df['string'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['synthesizer'] | df['synth']
df['synthesizer'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['violin'] | df['violins']
df['violin'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['vocal'] | df['vocals'] | df['voice'] | df['voices']
df['vocal'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['weird'] | df['strange']
df['weird'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')

new_column = df['soft'] | df['mellow']
df['soft'] = new_column
df.to_csv('annotations_131tags.csv',sep='\t')


