# %%
import re 

def remove_laughs(text):
    p = re.compile(r'\s(kk+)\s')
    text = p.sub(' ',text)
    p = re.compile(r'(kakkakakakkak)|(k+a+)+')
    text = p.sub(' ',text)
    p = re.compile(r'\W([hah?]{2,}|[heh?]{2,}|[hih?]{2,}|[huh?]{2,})\W?')
    text = p.sub(' ',text)
    p = re.compile(r'[\s]([hua]{2,}|[hue]{2,})[\s]')
    text = p.sub(' ',text)
    # p = re.compile(r'kk+')
    # text = p.sub(' ',text)
    return text

print(remove_laughs(' kkkkkkk brenda hahahhahahahaha hans huh '))

# %%
