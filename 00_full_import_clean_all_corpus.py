from constants import *

import pandas as pd
import re
import os
from io import open

from tqdm import tqdm


# In[2]:


if not os.path.exists(CORPUS.all_clean_files):
    os.makedirs(CORPUS.all_clean_files)


# In[3]:


all_corpus = {
    "corpus4encode_es-rev2021": {
        "check_dup": None,
        "to_keep": ["title", "indexTerm", "definition"]
    },
    "addCorpus": {
        "check_dup": "text",
        "to_keep": ["description", "text"]
    },
}


# In[4]:


def clean_def(d):
    startd = d + ""
    if d.startswith("b'") and d.endswith("'"):  # some wiki strings are b'...'
        d = eval(d).decode("utf-8")
    elif d.startswith('b"') and d.endswith('"'):  # some wiki strings are b"..."
        d = eval(d).decode("utf-8")
        
    
    if d.startswith("!markdown"):
        
        d = re.sub("\[(.+)\]\(.+\)", "\\1", d)   # markdown urls like [...](...)
        d = re.sub("\*\*", " ", d)               # markdown formatting **
        d = re.sub(r"\\\|", " |", d)             # markdown table columns \| --> |
        d = re.sub("\|([^|]*)(\s|$)", "\\1", d)  # markdown table columns |...|
        d = re.sub("(\+:?(-+|=+))+\+", " ", d)   # markdown table rows +---+ and +===+
        
        d = d.replace("!markdown", "")
    
    d = re.sub(";", " ; ", d)
    d = re.sub(" ;", ";", d)
    d = re.sub("\s+", " ", d)  # whitespace
    if len(d) == 0 and len(startd) != 0:
        print (startd)
    return d


# ### Final output for each corpus
# 
# ```python
# - corpus4encode-rev2020
#   "{title}. {definition}"
#     
# - corpusWiki
#   "{id}. {wikiSummary}"
# 
# - corpusWikiIndexTerm
#   "{indexTerm}. {wikiSummary}"
# 
# - corpusPubMed
#   "{id}. {pubmedTitle}. {pubmedAbstract}"
# 
# - corpusPubMedIndexTerm
#   "{id}. {pubmedTitle}. {pubmedAbstract}"
# ```

# In[5]:


final_all_clean_lines = []
set_final_all_clean_lines = set()
for (corpus, vals) in all_corpus.items():
    print(corpus)
    cols = vals["to_keep"]
    COL_NAME = vals["check_dup"]
    
    df = pd.read_csv(f"{corpus}.csv", low_memory=False)
    df = df.fillna("")
    #print(corpus)
    df = df[cols]
    # display(df.head(1))
    
    if COL_NAME is not None:
        print("dropping duplicates from column", COL_NAME)
        print("before:", df.shape[0])
        df = df.drop_duplicates()
        df = df[~(df[COL_NAME].duplicated(keep="first") & (df[COL_NAME] != ""))]
        print("after:", df.shape[0])
        print()
    
    clean_fields = []
        
    for i,col in enumerate(cols):
        clean_fields.append([])
        field = df[col].tolist()
        clean_field = [clean_def(f) for f in tqdm(field, desc=f"cleaning {col:<20}")]
        clean_fields[i] = clean_field   #lista di colonne pulite per ogni csv
        
    clean_lines = [str(f) for f in clean_fields[0]] 
    del clean_fields[0]
        
    while len(clean_fields) > 0:
        clean_lines = [f"{c}{'' if (len(c)>0 and c[-1]=='.') else '.'} {str(f)}" if len(f)>0 else c for c,f in zip(clean_lines, clean_fields[0])]
        del clean_fields[0]
            
    #print(len(clean_lines)+len(final_all_clean_lines))
    
    set_clean_lines = set()
    
    for l in clean_lines:
        set_final_all_clean_lines.add(l)
        if len(final_all_clean_lines)==0 or all([l not in s for s in final_all_clean_lines]):
            set_clean_lines.add(l)
    
    final_all_clean_lines.append(set_clean_lines)
    
    tot = 0
    for s in final_all_clean_lines:
        tot+=len(s)
    print("============================")
    print(tot)
    print(len(set_final_all_clean_lines))
    print("============================")
    


# In[7]:


for corpus, clean_lines in zip(all_corpus.keys(), final_all_clean_lines):
    print(corpus)
    
    if not os.path.exists(f"{CORPUS.all_clean_files}/{corpus}"):
        os.makedirs(f"{CORPUS.all_clean_files}/{corpus}")
    
    for idx,line in enumerate(tqdm(clean_lines)):
        with open(f"{CORPUS.all_clean_files}/{corpus}/{idx}.txt", "w", encoding="utf-8") as f:
            f.write(line)
    print()


# In[ ]:
