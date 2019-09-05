#%%
import pandas as pd
import math

#%%

os.chdir('/home/brenda/git/hate-speech/')

data = pd.read_csv('SentiLex-lem-PT02.txt',header=None)
df = data[0].str.split(';',n=1,expand=True)
df = df[0].str.split('.',n=1,expand=True)
data['expressao'] = df[0]
data['PoS'] = df[1]

df = data[0].str.split(':',n=1,expand=True)
df = df[0].str.split(';',n=1,expand=True)

data['TG'] = df[1]

df = data[0].str.split(';',n=2,expand=True)
df = df[2].str.split(';ANOT',n=2,expand=True)
data['POL'] = df[0]

df = data[0].str.split(';',n=3,expand=True)
data['ANOT'] = df[3]

df = data['POL'].str.split(';',n=1,expand=True)
data['POL'] = df[0]
data['POL2'] = df[1]

data['PoS'] = data['PoS'].str.replace('PoS=','')
data['TG'] = data['TG'].str.replace('TG=','')
data['ANOT'] = data['ANOT'].str.replace('ANOT=','')

data = data.drop(columns=[0])
data