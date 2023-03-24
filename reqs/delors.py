#!/usr/bin/env python
# coding: utf-8

# # Economic union in the debates on the creation of the euro: new evidence from the tapes of the Delors Committee meetings

# ## Authors

#  ### Emmanuel Mourlon-Druol [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/ORCID_ID) 
# European University Institute

# ### Enrico Bergamini [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-8930-6780) 
# University of Turin - Department of Economics and Statistics "Cognetti de Martiis"

# [![cc-by](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/) 
# ©<AUTHOR or ORGANIZATION / FUNDER>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY](https://creativecommons.org/licenses/by/4.0/)
# 

# [![cc-by-nc-nd](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/) 
# ©<AUTHOR or ORGANIZATION / FUNDER>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/)
# 
from IPython.display import Image, display

display(Image("./media/placeholder.png"))
#  

# ERC<br>
# WiSo workshop<br>
# Mathias Weber and Ross Higgins at ECB<br>

# text mining, delors, monetary union 

# Economic union is often said to have been overlooked in the negotiations on the creation of the euro. While there are many accounts of the government and central bankers’ negotiations to create the European single currency, the discussions within the Delors Committee which drafted a report outlining how to create an Economic and Monetary Union provide unique insights into the debates. Through a mixed-method analysis combining several text mining and network analysis techniques with a close reading of the transcripts, and a listening of the tapes, the article explores the debates that took place within the Committee. We identify which, when, by whom, and to what extent were different topics raised in the Committee, and find that economic union was the second most discussed topic during the Committee’s meetings, after the design and functioning of the future European central bank. The article further shows that the real absence from the debates was the question of banking regulation and supervision.

# ## Introduction

# A well-established criticism against Economic and Monetary Union (EMU) as created in 1992 is that it overlooked the economic dimension of monetary union (for example, <cite data-cite="14286898/YZW4P3TU"></cite>). How could the member states of the future monetary union compensate for the disappearance of the tool of exchange rate adjustment? Shouldn’t a greater European redistributive budget be created, a credible system for national economic policy coordination be put in place, and a coherent framework of banking regulation and supervision be set up? In spite of its name, EMU debates and negotiations leading up to the signature of the Maastricht Treaty would have sidelined ‘economic union’ to focus on monetary union alone. 
# 
# Among the debates that took place in the second half of the 1980s, the work of the so-called Delors Committee was instrumental in the making of EMU. The June 1988 European Council held in Hannover mandated a Committee chaired by President of the European Commission Jacques Delors to examine and propose concrete stages for the creation of a European Economic and Monetary Union (EMU). The Committee was composed of the twelve central bank governors of the European Economic Community (EEC) member states, the Commissioner in charge of agriculture Frans Andriessen, the General Manager of the Bank for International Settlements (BIS) Alexandre Lamfalussy, and the former Spanish finance minister Miguel Boyer. Two rapporteurs assisted the work of the Committee, Gunter Baer (BIS) and Tommaso Padoa-Schioppa (Banca d’Italia). The Committee met eight times, from September 1988 until April 1989. The Delors Committee published its final report on how to reach EMU, known as the Delors Report, in April 1989. The Delors Report made its way into the Maastricht Treaty with only a few minor changes.  
# 
# 
# 
# In this contribution, we examine closely the debates that took place in the Delors Committee, in order to better understand how its members reflected on ”economic union”.  With the benefit of hindsight, the Delors Committee and its Report have taken centre stage in analyses of the making of EMU, but the contemporaries’ impression was not that of having made history. It is famously recounted that when invited to write a paper in the Princeton International Economics Series, one of the rapporteurs, Padoa-Schioppa, turned down the offer as he doubted the report would become important <cite data-cite="14286898/WIFIUGXD"></cite>. However the Delors Report, for its greatest parts, made it through to the Maastricht Treaty; and its members — central bank governors and three experts — were central actors in the process. Being such an important milestone on the road to Maastricht, the work of the Delors Committee is frequently analysed through the lens of how the Committee members managed to find an agreement <cite data-cite="14286898/YZW4P3TU"></cite>. Central bankers were perceived as reluctant to EMU. In particular, the presence of two central bankers considered to be sceptical towards EMU, the president of the Bundesbank Karl-Otto Pöhl and the governor of the Bank of England Robin Leigh-Pemberton, further reinforces this view. Verdun examined whether the adoption of EMU in 1992 was rooted in consensus among monetary experts, specifically looking at the Delors Committee, and concluding that this Committee formed an epistemic community <cite data-cite="undefined"></cite>. 
# 
# Historians also mostly looked into this question from the same angle (<cite data-cite="14286898/88PNZ7P8"></cite>, <cite data-cite="14286898/PZ7Y5QYP"></cite>, <cite data-cite="14286898/PCBQZZ9M"></cite>, <cite data-cite="14286898/M7TTE5P4"></cite>, <cite data-cite="14286898/4AIUVZB6"></cite>, <cite data-cite="14286898/F4JPP4FL"></cite>). Harold James analysed how the members of the Delors Committee worked to find an agreement on the Report, making use of the records of the meetings, supported by some references to the tapes, in his analysis <cite data-cite="14286898/WIFIUGXD"></cite>).
# 
# By contrast, this article aims to analyse what the Committee actually discussed, rather than what the Committee published (the Delors Report). Our question is not so much “why EMU happened?” but is rather ”why has the Maastricht construct been so lopsided?” We now know why EMU happened; but the literature still overlooks why and how the ‘E’ of EMU has been neglected, or at least did not lead to a ‘symmetrical’ EMU. A closer analysis of the discussions that took place within the Committee can contribute to answer that question. Unusually for such gatherings, the meetings of the Delors Committee have been recorded on tape (47 hours of audible length), and partly transcribed word-for-word (about 90,000 words). The article uses these primary sources to analyse the discussions of the Committee, alongside the detailed inventory of the tapes, the rest of the archives of Committee, and the handwritten notes taken during the meetings by the two rapporteurs, Baer and Padoa-Schioppa <cite data-cite="14286898/XBZ8M5FL"></cite>. The article uses mixed methods, that is, a quantitative text analysis based on the meetings’ transcripts, alongside a qualitative analysis of the Committee’s discussions both transcribed and in the recorded tapes. Based on such a primary source, how can one evidence what the Delors Committee discussions were about? Reading the transcripts and listening to the tapes leaves little doubt about the fact that the Delors Committee members did alert multiple times about the consequences that irrevocably fixing exchange rates would have for the economic policy of the member states, and that the economic dimension of monetary union should be developed as the EEC/EU progresses towards the single currency. However, how can one translate this into historical evidence? Given the type of primary sources we used (tapes of the meeting, transcription of the records, preparatory notes, handwritten notes of the meetings) a mixed methods approach seemed to us the most appropriate. Unlike in the case of the Nixon tapes, for instance, we can use of the extensive word-for-word transcription of the debates <cite data-cite="14286898/49X25EH2"></cite>.
# 
# Quotations from the transcript, alone, would not convey the quantitative extent to which some topics were raised, and by whom. Conversely, numerical additions alone of what topic was raised when and by whom would fail to contextualise the quality and value of the issues at stake.In order to show how and to what extent the Delors Committee discussed policy issues related to ‘economic union,’ this article proceeds in three steps. At each step, we intertwined the narrative with an analysis of the hermeneutics behind and the data we use and present.
# 
# First, we explore who spoke about what during the Committee meetings, using entity recognition on the transcription of the Committee’s debates. Second, we analyse what topics were raised during the discussions, based on a topic modelling of the transcribed records, contrasted with references to the tapes that have not been transcribed. Third, we investigate what papers were distributed and not distributed to the Committee to further illustrate the substance of the Committee’s debates.

# ## Dataset construction

# In this paper, we present a novel dataset containing textual information from the recorded discussions that took place in the Delors Committee and that have been made available by the Historical Archives of the European Central Bank (available at: https://www.ecb.europa.eu/ecb/access_to_documents/archives/delors/html/index.en.html).
# 
# In opening the first meeting, Delors asked: 
# <br>
# 
# _“would you agree to the discussions being recorded and to being kept in great secrecy here at the BIS or are you against recording? I leave the decision to you. It is obvious that the rapporteurs will note the main points, but some things may escape their attention and it would be very helpful to them if we had a tape which they could refer to, a tape which would be kept here confidentially. Would you agree to this, Mr. Governors?”_
# 
# The governors agreed, and Delors added that _“You will be able to look them up [the records] and read them here. You can even listen to part of the tape if you wish, but here.”_ The records available are a word-for-word transcript of the tape that recorded the conversations in the Committee. It is however noticeable that the records contain gaps. Figure XXX provides the example of the very beginning of the first meeting of the Delors Committee. The numbers in brackets correspond to the reference points in the tapes. We can see that president of the Bundesbank Karl-Otto Pöhl spoke from 134 to 136, then Delors from 137 to 159, but that the next speaker in the page, governor of the Bank of England Robin Leigh-Pemberton, starts at 457. We can also see marks at the end of Delors’ intervention and at the beginning of Leigh-Pemberton’s intervention which imply that their interventions have not been transcribed in full. The interventions between points 160 and 456 of the tape are therefore missing. We will come back to this in the next section.

# In[1]:


from IPython.display import Image, display
display(Image("./media/fig1.jpg"))

print('Figure 1: Excerpt of the transcript of the tapes of the first meeting of the Delors Committee, 13 September 1988 available at https://www.ecb.europa.eu/ecb/access_to_documents/archives/delors/documents/shared/data/ecb.dr.delors880916_TranscriptionFirstMeeting.en.pdf?5d01ca39f8b12f73e7e476eb5986518a')


# In order to construct the dataset, we started by collecting all the scanned documents, containing the typewrote transcriptions of the meetings. In turn, we performed Optical Character Recognition (OCR) in order to convert the scans into machine-readable content. Since this process depends on the quality of the scans and is prone to generate typos and mistakes during the conversion, we manually corrected all discrepancies and errors in the digitisation process.  Thus, we obtained a dataset containing of 230 interventions, across the eight meetings. We manually curated the metadata for each intervention, such as: annotating speakers, speaking time, the manually checked and digitised text, and the reference to the tapes present in the original documents. 
# These references to the tapes allowed us to infer information about speaking time of each intervention, and also about the non-recorded parts of the discussions. In the next subsection, we provide further information into the process of reconstructing the extent of missing transcription, by leveraging speaking time and information obtained from the European Central Bank Archives.

# In[2]:


### code for performing OCR on ECB textual documents
#! cd scripts/OCR_IMAGES && python3 OCR_pdfs.py

### this files were in turn manually cleaned, translated, and aggregated into "script/delors.csv"


# ### Understanding missingness in transcriptions

# While being a novel and rich source of information, these transcripts are indeed not complete. How can we evaluate what is missing and what has been incompletely transcribed? The ECB archivist Matthias Weber established a detailed inventory at intervention level of the tapes. This means that each line of the 157 pages long inventory indicates when a member starts speaking and when he stops (by every second). In particular, Weber identified the transcription of the tape under three categories: complete, not included and abridged. The category abridged means that the intervention was not included in full, only in part, but it remained a word by word transcription. Based on this data, we have calculated that 10 hours and 30 minutes of recording have been transcribed in full, 2 hours and 15 minutes have been abridged in the transcription and 9 hours and 15 minutes are not included.

# In[3]:


import pandas as pd
from datetime import datetime
import numpy as np
from time import strftime, gmtime

import networkx as nx

import spacy
from multiprocess import Pool
from ast import literal_eval
from spacy.lang.en import English

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
get_ipython().run_line_magic('matplotlib', 'inline')

import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from tqdm.auto import tqdm
tqdm.pandas()

from string import punctuation
import re

from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim import corpora
import gensim

import tomotopy as tp

# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')


# In[4]:


i = 1

dfs = []
for sheet in ['13 Sep 88', '10 Oct 88', '08 Nov 88', '13 Dec 88', '10 Jan 1989', '14 Feb 1989', '14 Mar 1989', '11 April 1989', '12 April 1989']:
    d = pd.read_excel('./script/ECB_dataset.xlsx', sheet_name=sheet, skiprows=1)
    d['date'] = sheet
    d['meeting'] = i
    d = d[1:]
    dfs.append(d)
    i+=1
    
df = pd.concat(dfs)


# In[5]:


df = df.dropna(how='all', axis=1)
df = df[df['Speaker'].notna()]
df = df.reset_index(drop=True)


# In[6]:


def return_delta(s1, s2):
    try:
        if '.' in s1:
            FMT_s1 = '%M.%S'
        else: 
            FMT_s1 = '%M:%S'

        if '.' in s2:
            FMT_s2 = '%M.%S'
        else: 
            FMT_s2 = '%M:%S'

        tdelta = datetime.strptime(s2, FMT_s2) - datetime.strptime(s1, FMT_s1)
        return tdelta.total_seconds()
    except Exception as e:
        print(s1, s2, e)
        return np.nan

deltas = []
for s1, s2 in zip(df['Start of contribution'].tolist(), df['End of contribution '].tolist()):
    deltas.append(return_delta(s1, s2))
    
df['delta'] = deltas
df = df[df['delta'].notna()]


# In[7]:


df['duration'] = df['delta']


# In[8]:


df['meeting'] = df['meeting'].replace(9, 8)
df = df[df['duration']>=0]
df = df[df['Speaker']!='?']


# In[9]:


df['transcribed'] = df['In transcript'].str.replace(' $', '', regex=True)
df['transcribed'] = df['transcribed'].str.replace('Not$', 'Not included', regex=True)
df = df[df['transcribed'].isin(['Not included', 'Abridged', 'Complete'])]


# In[10]:


missing_transcriptions = pd.pivot_table(df, 
                          index='Speaker',
                          columns='transcribed',
                          values = 'duration',
                          aggfunc='sum',
                         ).round(0).fillna(0).reset_index().set_index('Speaker')

missing_transcriptions.loc['Total'] = missing_transcriptions.sum(numeric_only=True, axis=0)

for col in missing_transcriptions.columns:
    missing_transcriptions[col] = missing_transcriptions[col].apply(lambda x: strftime("%H:%M:%S", gmtime(x)))


# In[11]:


missing_transcriptions


# Why have some parts of the tapes been transcribed in full, some others in abridged version and finally some not at all? To try and answer that question we need to turn to the other major primary sources on the work of the Committee, namely, the handwritten notes of the two rapporteurs, Baer and Padoa-Schioppa. Based on these notes, we can deduce that what has been transcribed corresponds to what they perceived as the most important parts of the discussion. 
# 
# The analysis of the sets of handwritten notes of the two rapporteurs for each of the meetings confirms that they both knew, and indeed asked, for parts of the discussions to be transcribed. Both Baer and Padoa-Schioppa left asterisks in the margins of their handwritten notes that systematically correspond to those parts that have been transcribed. For instance, in his notes during the sixth meeting, Padoa-Schioppa explicitly wrote in red “transcript until Duisenberg” to mark the part he knew was transcribed. It is of course not possible to know whether some other persons intervened and requested the rapporteurs to ask for the transcriptions of some interventions specifically. Figure XXX provides another example, namely, page 5 of the handwritten notes of Baer for the third meeting of the Committee. Three successive speakers can be identified: governor of the Danish central bank Erik Hoffmeyer, governor of the National Bank of Belgium Jean Godeaux and Leigh-Pemberton. The latter is the only one marked with a cross, and indeed in the transcription of this third meeting, the first intervention recorded is that of Godeaux (but an earlier one), followed immediately by that of Leigh-Pemberton. But the tape numbers do not match: Godeaux’ intervention ends at 247 while that of Leigh-Pemberton starts at 589. This suggests that some interventions in between have not been recorded. We can however see that the handwritten notes of Baer where he put a cross in the margin (“agrees with Godeaux”) correspond to the transcription (“I must say that I align myself”).
# 

# In[12]:


display(Image("./media/fig2.jpg"))
print('Excerpt from the handwritten notes of Gunter Baer of the third meeting of the Delors Committee, 8 November 1988 available at https://www.ecb.europa.eu/ecb/access_to_documents/archives/delors/documents/shared/data/ecb.dr.delors881108_HandwrittenMinutesThirdMeeting.en.pdf?a673cf3af794467b87b0a7ecaa293877')


# In[13]:


display(Image("./media/fig3.png"))
display(Image("./media/fig4.png"))
print(' Excerpt of the transcript of the tapes of the third meeting of the Delors Committee, 8 November 1988 available at https://www.ecb.europa.eu/ecb/access_to_documents/archives/delors/documents/shared/data/ecb.dr.delors881110_TranscriptionThirdMeeting.en.pdf?bfd24e5eb630992f8a08521b3478b6ed')


# In[14]:


# - Who spoke for how long overall
speaker_overall = pd.pivot_table(df, 
                  index='Speaker',
                  values = 'duration', 
                  aggfunc='sum',
                 ).reset_index().sort_values(by='duration', ascending=False)

speaker_overall['duration'] = speaker_overall['duration'].apply(lambda x: strftime("%H:%M:%S", gmtime(x)))
# - And count of # of interventions per speaker per meeting
count_intervent = pd.pivot_table(df, 
                  index='Speaker',
                  columns='meeting',
                  values = 'duration', 
                  aggfunc='count',
                 ).reset_index().reset_index(drop=True)
# - Average length per meeting and per speaker (total length)
avg_intervent = pd.pivot_table(df, 
                  index='Speaker',
                  columns='meeting',
                  values = 'duration', 
                  aggfunc='mean',
                 ).round(0).fillna(0).astype(int)

for col in avg_intervent.columns:
    avg_intervent[col] = avg_intervent[col].apply(lambda x: strftime("%H:%M:%S", gmtime(x)))
    
avg_intervent = avg_intervent.reset_index().reset_index(drop=True)


# Having the detailed speaking time of each Delors Committee member allows us to uncover some of the dynamics of the Committee. Table XXX provides the ranking of the Delors Committee members by speaking time.

# In[15]:


speaker_overall


# Even before delving into an analysis of the substance of the Delors Committee members’ interventions, this ranking provides interesting preliminary indications. It is no surprise that Delors 8 hours 49 minutes), Pöhl (6 hours 55 minutes) and governor of the Banque de France Jacques de Larosière (5 hours 26 minutes), respectively chair of the Committee and governors of the two major central banks of the EEC spoke the most. 
# 
# Conversely, we notice that the two rapporteurs almost never intervened: their role, as rapporteurs, took place behind the scenes. It is however very surprising to find governor of the Central Bank of Ireland Maurice Doyle in the fourth position (3 hours 20 minutes), half an hour ahead of the governor of the Bank of England (2 hours 49 minutes) for instance. As governor of a Central Bank of a small state, Ireland, Doyle spoke much more than any of the other smaller member states: Tavares Moreira (Portugal) spoke for only 33 minutes, and Godeaux (Belgium), Mariano Rubio (Spain), Demetrios Chalikias (Greece), Pierre Jaans (Luxembourg) and Hoffmeyer (Denmark) for around an hour (50 minutes to 1 hour 16). This shows not only that the debates within the Committee did not necessarily follow the seniority or financial/political importance of the member states, but also that some topics outside the core aspects of monetary union were presumably tackled. 
# 
# Indeed, as we will show in greater detail below through the close-reading analysis, Doyle frequently raised the questions of wage issues (second meeting in particular) and financial transfers of resources to compensate for the loss of the ability to devalue. Doyle also submitted a specific paper to the Committee focusing on regional policy. Surprising is also the modest speaking time of Carlo Azeglio Ciampi (Italy), who - in spite of being governor of a large member state - spoke only 1 hour 39 minutes. Finally, we can notice that the three experts participating in the Committee have taken a full part in the discussion as they have all substantial speaking time, in particular Lamfalussy (5th speaker, 3 hours 10 minutes) and Boyer (7th speaker, 2 hours 37 minutes).
# 
# The calculation of the number of interventions and the average length of interventions confirms some patterns (see Tables XXX). With only one brief exception for Padoa-Schioppa in the third meeting, we can see that the rapporteurs’ interventions are strictly confined to the last meeting, which focused on the finalisation of the drafting of the Report. The average speaking time in this last meeting also highlights that all members intervened in the discussion, without significant discrepancies such as those observed in the overall speaking time. For instance, in this last meeting, Jaans, Hoffmeyer, Godeaux and Rubio all spoke on average longer than Delors, Pöhl and de Larosière. Overall, the relative shortness of average interventions indicates the flow of the debates as a genuine discussion, alternating longer presentations and multiple interjections.
# 

# In[16]:


avg_intervent.set_index('Speaker')


# While still partial, we now have a better overall view of who spoke when and for how long. From this information, we are confident that the data source we have constructed, despite not being the full transcript, reflects the core of the discussion. Having a word-by-word transcript of substantial size of what are the most important part of the tapes allows to use natural language processing tools to better identify the topics raised, while still being able to qualitatively interpret each individual intervention.
# In the next Section, we employ Named Entity Recognition (NER) in order to quantify cross-references between speakers, as well as references to institutions. In turn, we build networks of speaker-speaker interaction and speaker-institution interaction. In Section XXX we further detail the analysis by exploiting the digitized and clean text of the interventions with a machine learning approach, and infer latent topics present in the corpus.
# 

# ## The multiple guises of ‘economic union’: Analysis of the Delors Committee transcripts

# ### Who spoke about what in the Committee?

# In[17]:


d = pd.read_csv('./script/delors.csv')


# In[18]:


## clean and filter dataset

d['speaker_clean'] = d.speaker_clean.str.replace('Jaques', 'Jacques')

d = d[d['flag']!='missing']
d = d[d['text_english'].notna()]


# To understand who spoke about what in the Delors Committee meetings, we first worked on identifying speakers and entities from the dataset. We employed Named Entity Recognition (NER), in order to flag each word in the texts as a noun, an adjective, a verb, etc. Using this technique, we were able to extract (and manually normalize) the names mentioned in each intervention, and construct a network of mentions between speakers. We leveraged the SpaCy Python library in order to extract the relevant entities. We refer to the SpaCy documentation for further details on the specific types of entities that are possible to extract. In our case, we exploited the model in order to extract persons (proper names), and names of “companies, agencies, institutions, etcetera” (available at: https://spacy.io/usage/linguistic-features#named-entities )

# In[19]:


list_entities = ["PERSON",
                "NORP",
                "FAC",
                "ORG",
                "GPE",
                "LOC",
                "PRODUCT",
                "EVENT",
                "WORK_OF_ART",
                "LAW",
                "LANGUAGE",
                "DATE",
                "TIME",
                "PERCENT",
                "MONEY",
                "QUANTITY",
                "ORDINAL",
                "CARDINAL"
                ]


# In[20]:


### pointer to reproduce analysis or to re-read from saved csv file the NER data

run = False

if run == True:
    for label in list_entities:
        print(label)
        entities = []
        for t in d.text_english.tolist():
            try:
                doc = nlp(t)
                a = [ent.text for ent in doc.ents if ent.label_ == label]
                entities.append(a)
            except Exception as e:
                print(e)
                entities.append([])
        d[label] = entities
        
    d.to_csv('./script/delors_NER.csv')
else:
    d = pd.read_csv('./script/delors_NER.csv', converters={"ORG": literal_eval})


# ##### Named Entity Recognition
# 

# In[21]:


file_name = './script/networks/ner_entities_tot.xlsx'
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')   

for ind in list_entities:
    try:
        d1 = d[['meeting','speaker_clean', ind]]
        d1 = d1.explode(ind)
        p = pd.pivot_table(d1, index = ind, values = 'speaker_clean', aggfunc = 'count').reset_index()
        p = p.sort_values('speaker_clean', ascending=False)
        p.to_excel(writer, sheet_name=ind, startrow=0 , startcol=0, index=False)
    except Exception as e:
        print(e)
        
writer.save()


# ##### Per meeting

# In[22]:


file_name = './script/networks/ner_entities.xlsx'
writer = pd.ExcelWriter(file_name, engine='xlsxwriter')   

for ind in list_entities:
    try:
        d1 = d[['meeting','speaker_clean', ind]]
        d1 = d1.explode(ind)

        aa = pd.DataFrame()
        p = pd.pivot_table(d1, index = ind, values = 'speaker_clean', columns= 'meeting', aggfunc = 'count').reset_index()

        for i in range(1,9):
            p = p.sort_values(by=i, ascending=False)
            aa[i] = p[ind].tolist()[:20]
            aa[str(i)+'_count'] = p[i].tolist()[:20]
        aa.to_excel(writer, sheet_name=ind, startrow=0 , startcol=0, index=False)
    except Exception as e:
        print(e)
        
writer.save()


# ### Network analysis

# In[23]:


n = d[['speaker_clean', 'ORG']]
n = n.explode('ORG')

G = nx.from_pandas_edgelist(n, 'speaker_clean', 'ORG')
print("Graph has "+str(len(G.nodes))+" nodes and "+str(len(G.edges))+" edges")


# In[24]:


def draw(G, pos, measures, measure_name, sizes, labels=False):
    nodes = nx.draw_networkx_nodes(G, pos, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys(),
                                   node_size=sizes,
                                  )
    
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    if labels == True: 
        labels_dict = {}
        for node, size in zip(G.nodes, sizes):
            deg = size / 100
            if deg >= 5:
                labels_dict[node] = node
        nx.draw_networkx_labels(G,pos,labels_dict,font_size=16,font_color='black')
 
    edges = nx.draw_networkx_edges(G, pos)
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()

nx.write_gexf(G, "./script/networks/graphs/ORG.gexf")
n.rename(columns={'speaker_clean':'Source', 'ORG':'Target'}).to_csv('./script/networks/graphs/ORG_data.csv', index=False)

# the gexf data was styled in Gephi, producing the following Figure:


# In[25]:


display(Image("./media/fig5.png"))

print('Bipartite network of the entities and speakers')


# Figure XXX shows the bipartite network representing speakers (grey nodes) and entities (white nodes) mentioned in the meetings. The speakers are the members of the Delors Committee; the entities are any organisation and notion that they mentioned (an institution, a report or a policy for instance). Each entity mentioned in the transcription of the records is linked to a different intervention of our dataset, and can therefore be attributed to each speaker. In the network, the edges represent a directional relationship between the nodes. If Jacques Delors mentions the Bank for International Settlements in an intervention, the edge will represent an arrow going from the node ‘Jacques Delors,’ and aiming at the node ‘Bank for International Settlements.’ If Delors has mentioned the BIS more times than the Council of Ministers, the edge towards the BIS will have a heavier weight. The weights of the edges are simple sums of the mentions. To make it readable, the figure excludes entities and speakers that appeared only once.
# Looking at the entities to which the Delors Committee members referred, there is little trace of economic union. We can only see a marginal mention of the Centre for Economic Policy Coordination (an institution at a time envisaged as the economic authority counterbalancing the future European monetary authority) as well as references to the 1974 Convergence Directive and the European Fiscal Framework. The entities recognised are predominantly institutions, either from the European Economic Community (European Council, European Parliament, Council of Ministers…) or from member states (national central banks). As the next sections will show, the absence of economic union-related entities does not mean however that economic union-related topics were not discussed. This absence rather shows that their discussion did not revolve around well-identified concepts. Instead, discussion of economic union implied broader conversations about how economic and monetary policies interact, that cannot be reflected and identified through the mention of a few standard entities. 
# 

# The institutions associated with Pöhl confirm that his main concerns centred around the ECU and the EMS. The role of the ECU, its possible use as a parallel currency, were indeed key preoccupations of the Bundesbank. Reference to the EMS denotes the reference to the current system by contrast to what could be done, and highlights a specific reflection on the advantages and disadvantages of moving to something new, that was accompanied by a paper he submitted to the Committee (Karl-Otto Pöhl, The further development of the EMS, 14 September 1988, see ECB Archives, CSEMU, List of papers by subject, 3 January 1989).
# 
# We can also see that Pöhl is the central banker who most often referred to his own central bank, the Bundesbank. This underscores the peculiar status (full independence) of the Bundesbank: Pöhl had to regularly set out clearly his own domestic institutional constraints to the other members of the Committee. For instance, during the third meeting Pöhl elaborates on the different institutional options for the central bank and refers to the independence of the Bundesbank, and in the sixth meeting Pöhl explains the board of the Bundesbank's appointment procedure to the members of the Committee. The main topics associated with Delors highlight his role as Committee chair and president of the Commission. The EEC is the entity he most often referred to, followed by the ECU (one of the options on the table of discussions), the Committee itself (what it is doing, how it functions), and the European Council (the institution giving the mandate to the Committee). 
# Apart from these two cases, the entities most referred to are well distributed. Of course, references to both the Delors Committee and the Delors Report stand out, since they were the purpose of the meetings. The European Reserve Fund was one of the main elements of discussion, and an idea put forward by de Larosière in a paper submitted to the Committee. The figure underlines however that the ERF was equally referred to by all members, not just by de Larosière, which highlights a genuine debate among the members.
# 

# ### Which words were the most used?

# In[26]:


d = pd.read_csv('./script/delors_NER.csv')


# In[27]:


## Remove stopwords in english
stopwords = list(nltk.corpus.stopwords.words('english'))
ad_hoc_sw = ['from', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'dont', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'line', 'also', 'may', 'take', 'come', 'one', 'amp', 'well']
extensions = ['mr', 'the', 'u', 'mr', 'I', 'would', 'think', 'say', 'also', 'may', 'go', 'make', 'can', 'not']
stopwords.extend(ad_hoc_sw)
stopwords.extend(extensions)

# Stemming
ps = nltk.stem.SnowballStemmer('english')
# Lemmatization
wn = nltk.WordNetLemmatizer()
# Polishing the text

'''
Here are a set of functions to clean and preprocess english text. 
The first one should take care of most of the issues, but subsequently I apply more further on
To deactivate some functions comment out last block
'''

def remove_punct(text):
    text = re.sub('(?:\@|https?\://)\S+', '', text) #removes accounts mentioned and urls
    text  = "".join([char for char in text if char not in punctuation])
    text = re.sub('[0-9]+', '', text)
    text = re.sub('â', "'", text) #some weird shit
    return text.lower()

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    text = [word for word in text if word not in stopwords and len(word) > 1]
    return text

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

def lemmatizer(text):#, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(text)) 
    lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha and tok.text.lower() not in stopwords]
    return text

def non_vec(text):
    text = ' '.join(word for word in text)
    return text
d['text_clean'] = d['text_english'].progress_apply(remove_punct)
d['text_clean'] = d['text_clean'].progress_apply(tokenization)
d['text_clean'] = d['text_clean'].progress_apply(remove_stopwords)
d['text_clean'] = d['text_clean'].progress_apply(lemmatizer)
d['text_clean'] = d['text_clean'].progress_apply(non_vec)


# In[28]:


### Only more than 3 words
d['len'] = d['text_clean'].str.split().apply(len)
d = d[d['len']>3]
speakers = list(d.speaker_clean.unique())
speakers.remove('?')


# In[29]:


by_meeting_words = pd.DataFrame()
by_meeting_bigrams = pd.DataFrame()

for i in range(1,8+1):
    try:
        d1 = d[d['meeting']==i]

        tokens = nltk.tokenize.word_tokenize(' '.join(d1.text_clean.tolist()))
        #Create your bigrams
        bgs = nltk.bigrams(tokens)
        bigram_fdist = nltk.FreqDist(bgs)
        words_fdist = FreqDist(tokens)
        
        top_words = [w + '  ' +str(c) for w, c in words_fdist.most_common(10)]
        top_bgs = [w[0] + '_' + w[1] + '  ' + str(c) for w, c in bigram_fdist.most_common(10)]
        
        by_meeting_words[i] = top_words
        by_meeting_bigrams[i] = top_bgs
        
    except Exception as e:
        print(e)
        print(i)
        


# In[30]:


by_meeting_bigrams


# Another door of entry to explore the dataset and identify who spoke about what and when is to analyse the words most used. In this section, we focus on very simple metrics, counting the appearance of expressions in the text of the interventions, while in the next section we will rely on more refined Natural Language Processing techniques.
# We privileged analysis of bigrams (that is, pairs of consecutive words) over analysis of words alone, as the subject studied has an inherent bias towards words taken separately. For instance, ‘economic and monetary union’ would result in counting three words for one concept alone, and thus not inform much about the distribution of topics and themes in the discussions. The three most used single words of Delors throughout the transcripts are indeed ‘economic’ (103 occurrences), ‘monetary’ (86) and ‘union’ (84).
# Turning to the bigrams, the top bigram for each meeting is always ‘monetary union’ or ‘central bank,’ with the exception of meeting 7, when the top bigram is ‘political signal’. This reflects the fact that discussions in the seventh meeting focused on the impact different draftings of the Report could have on the European Council. Overall, ‘monetary union’ is unsurprisingly the most frequently used bigram (181 occurrences, bearing in mind that this figure is biased by the fact that the oft-used phrase ‘economic and monetary union’ will only be recorded in terms of bigram as ‘monetary union’), followed by ‘monetary policy’ (111), ‘central bank’ (102), ‘economic monetary’ (98, most likely a reference to EMU), ‘exchange rates’ (68), and finally ‘economic union’ (48). Decomposing the bigrams per speaker, we observe that ‘economic union’ is referred to by the two experts, Boyer and Thygesen, and by Delors. This time, some economic union-related topics reflected in other bigrams emerge. Doyle refers to the wages implications and Tavares Moreira to the regional implications of EMU. Bigrams thus offer a first glimpse into the presence, in the background, of a discussion on “economic union.”
# 
# While offering interesting insights about the substance of the discussions, the bigrams analysis and the deductions we can draw from the speaking times are limited. Many items that could be included in the broad theme of economic union, such as fiscal policy, regional policy and coordination of economic policies, cannot be solely reflected under the bigram of ”economic union”. Conversely, the bigram of ”economic union” cannot be unpacked to understand what exactly were the speakers referring to. In order to compensate for that limitation, we employ a machine learning approach and apply topic modelling in order to infer topics from the corpus.

# In[31]:


by_speaker_words = pd.DataFrame()
by_speaker_bigrams = pd.DataFrame()

for i in speakers:
    try:
        d1 = d[d['speaker_clean']==i]

        tokens = nltk.tokenize.word_tokenize(' '.join(d1.text_clean.tolist()))
        #Create your bigrams
        bgs = nltk.bigrams(tokens)
        bigram_fdist = nltk.FreqDist(bgs)
        words_fdist = FreqDist(tokens)

        top_words = [w + '  ' +str(c) for w, c in words_fdist.most_common(10)]
        top_bgs = [w[0] + '_' + w[1] + '  ' + str(c) for w, c in bigram_fdist.most_common(10)]
    #     print(top_words)
    #     print(top_bgs)

        by_speaker_words[i] = top_words
        by_speaker_bigrams[i] = top_bgs
        
    except Exception as e:
        print(e)
        print(i)
        


# In[32]:


by_speaker_bigrams


# ### What topics were raised?

# We move beyond simple word-counts by inferring latent topics in the texts, employing a topic modelling approach. Starting from the dataset of full-text interventions, we then refined the text with the standard techniques of pre-processing. We removed stop words (very common words like “the”) and punctuation. We also lemmatised each word, that is, we brought conjugations and declinations to their lemma (e.g. the word “going” is mapped as to “go”). We also add to the model bigrams, hence expressions which are often in pairs. For instance, the words “monetary” and “policy” would be tied together in “monetary_policy”. Furthermore, given that interventions have very different structures and lengths (e.g. a quick reply or a long statement) we split the text into sentences rather than having the full text of the intervention. 
# 
# The rationale for doing this is also inherent into the aim of our exercise: each intervention can blend a wide variety of topics, and hence cannot be solely allocated to one topic. For instance, one speaker can start by tackling the issue of the role of the ECU, and then move on to talk about the economic implications of fixed exchange rates, and conclude on the importance of having a political impulse for monetary union to materialise. Splitting each of the 230 long interventions by sentences can help us cluster better the content of the speeches into topics. Therefore, we construct a dataset representing a corpus of 3043 sentences with an after cleaning dictionary of 4395 unique words and bigrams. 
# 
# We choose to implement a Latent Dirichlet Allocation unsupervised model, in order to infer latent topics from the documents, and assign them to specific topics. Relying only on keyword matching, as mentioned, is limited as it misses the latent component that these models instead bring. Our aim is to cluster sentences and interventions into topics and sub-topics independently of our pre-conceived notion of the topics discussed during the meetings. While unsupervised machine-learning allows us to do so, it also comes with limitations. The use of language during those meetings is that of high-level policymakers into formal meetings, and clustering correctly the content of interventions might not be a trivial in and unsupervised-learning framework. 
# 
# Hence, we aim at leaving flexibility to the model, while at the same time nudging it to recognize specific topics without resorting to a rigid set of keywords matches. For these reasons, we resort to the guided-LDA model, an extension of the base LDA algorithm that allows us to insert lexical priors into the model. These lexical priors are a set of pre-determined keywords describing topics that, a-priori, we believed were present in the corpus. We defined the prior topics based on the close-reading of the documents and are presented in Table XXX.  The topics seeded into the models as priors, nudge it to recognise them and help us to cluster more efficiently the documents into specific topics. At the same time, contrary to keywords, we still exploit a stochastic component: the model might converge into recognizing topics that are not in our list of priors, or into not recognizing seeded topics. 

# Table XXX: seeded topics in the guided-LDA model
# 
# |Topic|Seeded words|
# |:----|:----|
# |0 organisational aspects|baer, minutes, record, week, unanimity, compromise, statement|
# |1 parallelism of progress between economic union and monetary union|pace, force, monetary_union, economic_union, werner, step, precondition, speed, parallel, parallelism|
# |2 definition of monetary union|irreversible, irrevocable, fixed, balance, flexible, single_currency, common_currency, sovereignty, sovereignty, margin, narrow, narrowing, locked|
# |3 economic union|internal, market, regional, fiscal_policy, common_policy, accompanying_policies|
# |4 outline of report and drafting discussions|step, treaty, change, stage, first, second, transition, scenario, phase, final, signal, impulse, negotiate, credibility, compromise, paragraph, process, statement,sentence|
# |5 regional policy|region, regional_policy, policy_regional, regional, exchange_rate, transfer, common_policy, transfers, peripheral, central, difference, infrastructure, regional_transfers, imbalance, imbalances, adjustment, structural, regional_development,|
# |6 fiscal policy|transfer, budget, budgetary policy, common_policy, adjustment, financial, increase, community_budget|
# |7 ECU and its role|parallel_currency, private, private_ecu, ecu, currency, basket, attractive, business|
# |8 design of ECB|independence, board, decision, rate, interest rate, authority, central, credibility, european_central_bank, central, law, escb, administer, third_currencies, council_governors, quantitative, control, technique, aggregates, reserves, compulsory|
# |9 European Reserve Fund|larosière, monetary_fund, erf, reserve_fund|
# |10 economic policy coordination|convergence, exchange_rate, capital, adjustment, directive, coordination, welfare, centre, precondition, binding, inflation|
# |11 cost of production and competitiveness|wages, exchange_rate, income, unemployment, employment, productivity, adjustment, competitive, work, wage, migration|
# |12 tax hamonisation|taxes, exchange_rate, harmonise|
# |13 institutional structure|council, ecofin, parliament, european_parliament, statement, veto, institution, sovereignty, brussels, central, centralisation, law, speed, decision, fecom, credibility, authority, european_council, heads_state, heads_government, subsidiarity, rome, treaty_rome, legitimisation|
# |14 budget deficit|market, market_mechanism, constraint, borrow, limit, gdp, savings, binding, size, discipline|

# Table XXX shows the composition of the dataset per meeting.
# 
# |Meeting number|Date of meeting|Number of words of the transcript|Number of pages|
# |:----|:----|:----|:----|
# |1st meeting|13 September 1988|8091|20|
# |2nd meeting|10 October 1988|17096|41|
# |3rd meeting|8 November 1988|9336|21|
# |4th meeting|13 December 1988|7849|20|
# |5th meeting|10 January 1989|21165|49|
# |6th meeting|14 February 1989|7742|20|
# |7th meeting|14 March 1989|7563|18|
# |8th meeting|12 April 1989|10469|24|
# |Total| |89311|213|

# We apply the guided LDA to the pre-processed and clean version of the text. The outcome of these models are a set of keywords for each topic, that in turn have to be interpreted. Our model stabilizes evaluation metrics (coherence) around 70 topics. Each sentence could belong to each of these topics according to a probability distribution. As common practice, we assign the sentence to the topic for which it has the highest probability. 

# In[33]:


nlp = spacy.load('en_core_web_sm')

# Up and clean
d = pd.read_csv('./script/delors_NER.csv')
d['speaker_clean'] = d.speaker_clean.str.replace('Jaques', 'Jacques')
d = d[d['flag']!='missing']
d = d[d['text_english'].notna()]


# In[34]:


nlp = English()  # just the language with no model
nlp.add_pipe('sentencizer')

split_in_ = 'sents' #or 'paragraphs', 'nothing', 'sents'

## SpaCy with sentencizer
if split_in_ == 'sents':
    d["text_split"] = d["text_english"].apply(lambda x: [sent.text for sent in nlp(x).sents])
    d = d.explode('text_split')
    
elif split_in_ == 'paragraphs':
    def split_paragraphs(text):
        sents = re.split('\n\n|\.\.\.', text)
        #sents = re.split('\.', text)
        return sents
    d['text_split'] = d['text_english'].apply(lambda x: [sent for sent in split_paragraphs(x)])
    d = d.explode("text_split")
elif split_in_ == 'nothing':
    d['text_split'] = d['text_english']
d['text_toclean'] = d['text_split']


# In[35]:


nlp = spacy.load('en_core_web_sm')
## Remove stopwords in english
stopwords = nltk.corpus.stopwords.words('english')
ad_hoc_sw = ['from', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'dont', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'line', 'also', 'may', 'take', 'come', 'one', 'amp', 'well']
extensions = ['mr', 'u']
stopwords.extend(extensions)
stopwords.extend(ad_hoc_sw)

# Stemming
ps = nltk.stem.SnowballStemmer('english')
# Lemmatization
wn = nltk.WordNetLemmatizer()


# In[36]:


# Polishing the text

'''
Here are a set of functions to clean and preprocess english text. 
The first one should take care of most of the issues, but subsequently I apply more further on
To deactivate some functions comment out last block
'''

def remove_punct(text):
    text = re.sub('(?:\@|https?\://)\S+', '', text) #removes accounts mentioned and urls
    text  = "".join([char for char in text if char not in punctuation])
    text = re.sub('[0-9]+', '', text)
    text = re.sub('â', "'", text) #some weird shit
    return text

def tokenization(text):
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    text = [word for word in text if word not in stopwords]
    return text

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

def lemmatizer(text):#, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(text)) 
    lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha and tok.text.lower() not in stopwords]
    return text

def non_vec(text):
    text = ' '.join(word for word in text)
    return text

d['text_clean'] = d['text_toclean'].progress_apply(remove_punct)
d['text_clean'] = d['text_clean'].progress_apply(tokenization)
d['text_clean'] = d['text_clean'].progress_apply(remove_stopwords)
d['text_clean'] = d['text_clean'].progress_apply(lemmatizer)
d['text_clean'] = d['text_clean'].progress_apply(non_vec)


# In[37]:


## Remove stopwords in english
stopwords = nltk.corpus.stopwords.words('english')

corpus_texts = d.text_clean.tolist()
corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=lambda x: len(x) <= 1 or x in stopwords)
corpus.process([text for text in corpus_texts])

# extract the n-gram candidates first
cands = corpus.extract_ngrams(min_df=10, max_len=5, max_cand=200)
print('==== extracted n-gram collocations ====')

# concat n-grams in the corpus
corpus.concat_ngrams(cands, delimiter='_')


# In[38]:


# Guided LDA with seed topics.
priors_dict = {
        0: ["baer", "minutes", "record", "week", "unanimity", "compromise", "statement"],
        1: ["pace", "force", "monetary_union", "economic_union", "werner", "step", "precondition", "speed", "parallel", "parallelism"],
        2: ["irreversible", "irrevocable", "fixed", "balance", "flexible", "single_currency", "common_currency", "sovereignty", "sovereignty", "margin", "narrow", "narrowing", "locked"],
        3: ["internal", "market", "regional", "fiscal_policy", "common_policy", "accompanying_policies"],
        4: ["step", "treaty", "change", "stage", "first", "second", "transition", "scenario", "phase", "final", "signal", "impulse", "negotiate", "credibility", "compromise", "paragraph", "process", "statement","sentence"],
        5: ["region", "regional_policy", "policy_regional", "regional", "exchange_rate", "transfer", "common_policy", "transfers", "peripheral", "central", "difference", "infrastructure", "regional_transfers", "imbalance", "imbalances", "adjustment", "structural", 'regional_development'],
        6: ["transfer", "budget", "budgetary policy", "common_policy", "adjustment", "financial", "increase", "community_budget"],
        7: ["parallel_currency", "private", "private_ecu", "ecu", "currency", "basket", "attractive", "business"],
        8: ["independence", "board", "decision", "rate", "interest rate", "authority", "central", "credibility", "european_central_bank", "central", "law", "escb", "administer", "third_currencies", "council_governors", "quantitative", "control", "technique", "aggregates", "reserves", "compulsory"],
        9: ["larosière", "monetary_fund", "erf", "reserve_fund"],
        10: ["convergence", "exchange_rate", "capital", "adjustment", "directive", "coordination", "welfare", "centre", "precondition", "binding", "inflation"],
        11: ["wages", "exchange_rate", "income", "unemployment", "employment", "productivity", "adjustment", "competitive", 'work', 'wage', "migration"],
        12: ["taxes", "exchange_rate", "harmonise"],
        13: ["council", "ecofin", "parliament", "european_parliament", "statement", "veto", "institution", "sovereignty", "brussels", "central", "centralisation", "law", "speed", "decision", "fecom", "credibility", "authority", "european_council", "heads_state", "heads_government", "subsidiarity", "rome", "treaty_rome", "legitimisation"],
        14: ["market", "market_mechanism", "constraint", "borrow", "limit", "gdp", "savings", "binding", "size", "discipline"],
}

seed_topic_list = [priors_dict[i] for i in priors_dict]


# In[39]:


def search_gLDA(corpus, list_topic_nums, alphas, iter_num, min_df=10, min_cf=5):
    cv_values = []
    umass_values = []
    model_list = []
    alphas_out = []
    
    for alpha in tqdm(alphas):
        for num_topics in tqdm(list_topic_nums, leave=False):
            # print("Training: {} alpha and {} topics...".format(alpha, num_topics))
            ### Build model
            mdl = tp.LDAModel(k=num_topics, 
                  min_cf=min_df, 
                  min_df=min_cf, 
                  corpus=corpus,
                  alpha=alpha
                 )
            
            ## Seed topics
            for seed_id in priors_dict:
                for prior_word in priors_dict[seed_id]:
                    mdl.set_word_prior(prior_word, [0.95 if k == seed_id else 0.1 for k in range(num_topics)])
            
            # init
            mdl.train(0)
            
            # train
            for i in range(0, iter_num, 10):
                mdl.train(10)
                # print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

            ### get stats
            model_list.append(mdl)
            coh_cv = tp.coherence.Coherence(mdl, coherence='c_v')
            coh_um = tp.coherence.Coherence(mdl, coherence='u_mass')
            cv_values.append(coh_cv.get_score())
            umass_values.append(coh_um.get_score())
            alphas_out.append(alpha)

    return model_list, cv_values, umass_values, alphas_out


# In[40]:


### List of hyperparameters
iter_num = 500 
list_topic_nums = np.arange(10, 200, 10).tolist()
alphas = [0.0001, 10]
re_run = False

if re_run == True:
    ## Search wrapper func
    model_list, coherence_values, umass_values, alpha_values = search_gLDA(
                                                            corpus = corpus,
                                                            list_topic_nums = list_topic_nums, 
                                                            iter_num = iter_num,
                                                            alphas = alphas,
                                                            min_df = 20,
                                                            min_cf = 10,
                                                            )
    


# In[41]:


import numpy as np
import matplotlib.pylab as plt

if re_run == True:
    ### Create DF for all the models
    val = pd.DataFrame()
    val['Num Topics'] = [model_list[i].k for i, v in enumerate(model_list)]
    val['Coherence Score'] = coherence_values
    val['Alpha Parameter'] = alpha_values
    val['Alpha Parameter'] = val['Alpha Parameter'].fillna(0)
    val['UMass Score'] = umass_values
    val['Models'] = model_list

    measures = ['Coherence Score', 'UMass Score']

    for measure in measures:
        v = pd.pivot_table(val, index='Num Topics', values=measure, columns='Alpha Parameter')
        v = v.reset_index().reset_index(drop=True)

        ys = []
        for alpha in alphas:
            v = val[val['Alpha Parameter']== alpha]
            ys.append(v[measure].values)

        x = list_topic_nums
        plt.figure(figsize=(10, 6), dpi=100, facecolor='w', edgecolor='k')
        n = len(alphas)
        colors = plt.cm.Blues(np.linspace(0.5,1,n))

        for i, a in zip(range(n), alphas):
            y = ys[i]
            plt.plot(x, y, color=colors[i], label=a)

        plt.xlabel("Num Topics")
        plt.ylabel(measure)
        plt.title("Delors Dataset")
        plt.legend(loc='lower right', 
                    shadow=False, 
                    title='Alpha', 
                    bbox_to_anchor=(1.15, 0)
                    )

        plt.savefig('./script/topic_model/gLDA_{}_{}.png'.format('mallet', measure), dpi=200)
        plt.show()


# In[42]:


optimal_alpha = 10

if re_run == True:
    vv = val[val['Alpha Parameter']==optimal_alpha]
    print(vv.loc[vv['Coherence Score'].idxmax()])
    print('\n')
    print(vv.loc[vv['UMass Score'].idxmin()])
    #vv.head()
optimal_topics = 70
optimal_alpha = 10

if re_run == True:
    model = val[(val['Alpha Parameter']==optimal_alpha) & (val['Num Topics']==optimal_topics)]['Models'].values[0]
    model.save('./script/topic_model/gLDA.bin')
else:
    model = tp.LDAModel.load('./script/topic_model/gLDA.bin')
for k in range(model.k):
    print("== Topic #{} ==".format(k))
    for word, prob in model.get_topic_words(k, top_n=10):
        print(word, prob, sep='\t')
    print()


# In[43]:


mdl = model
topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq


# In[44]:


# extract candidates for auto topic labeling
extractor = tp.label.PMIExtractor(min_cf=5, max_cand=1100)
cands = extractor.extract(mdl)

labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
# for k in range(mdl.k):
#     print("== Topic #{} ==".format(k))
#     print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=10)))
#     for word, prob in mdl.get_topic_words(k, top_n=10):
#         print(word, prob, sep='\t')
#     print()

topics_labels_df = pd.DataFrame()
topics_labels_df['topics'] = np.arange(0,mdl.k,1)
words_ = [', '.join(word for word, prob in mdl.get_topic_words(k, top_n=10)) for k in range(mdl.k)]
labels_ = [', '.join(label for label, score in labeler.get_topic_labels(k, top_n=10)) for k in range(mdl.k)]
topics_labels_df['labels'] = labels_ 
topics_labels_df['words'] = words_ 

topics_labels_df.to_excel('./script/topic_model/topic_labels.xlsx', index=False)


# In[45]:


doc_topics_df = pd.DataFrame(doc_topic_dists)

dominant_topics = []
max_probabilities = []

# Get the indices of maximum element in numpy array
for row in doc_topic_dists:
    max_prob = maxElement = np.amax(row)
    dom_topic = np.where(row == np.amax(row))[0]
    dominant_topics.append(dom_topic)
    max_probabilities.append(max_prob)


# In[46]:


# create
df_dominant_topic = pd.DataFrame(columns=['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords'])

# assign
df_dominant_topic['Document_No'] = range(0,len(doc_topic_dists))
df_dominant_topic['Dominant_Topic'] = [dt[0] for dt in dominant_topics]
df_dominant_topic['Topic_Perc_Contrib'] = max_probabilities
df_dominant_topic['Keywords'] = [topics_labels_df.at[k[0], 'labels'] for k in dominant_topics]


# In[47]:


def get_len_sent(x):
    if x:
        return len(x.split())
    else:
        return 0


# In[48]:


d['len_sentence'] = d['text_clean'].apply(get_len_sent)

d[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']] = df_dominant_topic[['Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]


# In[49]:


d.to_csv('./script/delors_gLDA.csv', index=False)
d.to_excel('./script/topic_model/delors_gLDA.xlsx', index=False)


# In turn, we manually interpret and label the 70 topics into two sets of hierarchical labels. As subtopics, where we only interpret the results of the models, and as higher-level topics. In table XXX we present the results of the topic modelling methodology, as well as the two sets of labels we defined.

# In[50]:


## re-read labelled dataset
pd.options.display.max_rows = 200
tlabel = pd.read_excel('./script/topics_manually_labelled.xlsx', 
                       sheet_name='guidedLDA',
                       names=['topics', 'labels', 'words', 'manual_label', 'manual_label_parent', 'comment'],
                       usecols = ['topics', 'labels', 'words', 'manual_label']
                      )
tlabel


# Finding the coherent theme that unifies the topics is the most difficult interpretative task. A close reading of the interventions grouped under the eight higher-level topics identified by the LDA however confirms the automated results. Table XXX below provides a descriptive title to each of the topics identified in the LDA. 

# |Topic 0|EMU design|
# |:----|:----|
# |Topic 1|Economic union|
# |Topic 2|Monetary policy|
# |Topic 3|ECU|
# |Topic 4|Institutional change|
# |Topic 5|Miscellaneous|
# 
# Table XXX: Analytical presentation of the gLDA topics

# Topic 0 concerns EMU design, and is the most prominent of all. It concerns issues related to the setting up of scenarios, the general structure of EMU, reflections on the advantages and drawbacks of EMU, as well as drafting questions; topic 1 concerns ‘economic union’ broadly speaking, that is, how will member states cope with the inability to devalue their currency, and therefore includes the role of fiscal policy, regional policy and wages; topic 2 relates more specifically to monetary union and especially the future common monetary policy; topic 3 is on the ECU and its use; topic 4 relates to institutional change, that is, the need for a new treaty, the question of the political impulse, and of the overall institutional balance; and finally topic 5 covers sentences that do not really relate to a specific topic, but are instead general comments, sentences of transition between statements, or simply unrelated to work of the Committee. Looking at the grand totals, we can see that most of the discussion focused on the first two topics, EMU design and economic union, that add up to almost exactly three quarters of the total (75.9%). The other topics are very little discussed overall. 
# Unpacking the first two topics which are the two most relevant for our argument allows to further refine the picture. Topic 0 on EMU design encompasses very broad reflections about EMU, including some related to the purpose of the Report. For instance, Doyle’s description of what the Committee is doing matched topic ‘EMU design’: “the purpose of this Report basically is to go back to the European Council and say: you said you wanted EMU, do you really understand what it implies and that we have set out, and says then, alright if you still want it then this is how you go about it.” Topic 0 also includes reflections on the balance between economic union and monetary union – therefore touching upon Topic 1 – and the issue of the so-called parallel progress between both. For instance, Leigh-Pemberton’s call for parallel progress in monetary and economic integration – “I am firmly of the view that if in fact progress is to be made it has got to be parallel progress with monetary and exchange rate co-operation on the one side and economic progress on the other” – ends up being associated with the topic of ‘EMU design’.  Similarly, Thygesen’s reflections on the balance between economic and monetary integration in the Report, concluding that he is happy with the report as it stands in spite of the imbalance in favour of monetary integration. However, other remarks by the same actors during the first meeting give opposite categorisation, with Thygesen ending in the ‘economic union’ topic (as he reviews progress since the snake, and notes that monetary integration moved ahead of economic integration) while Leigh-Pemberton ends in the ‘EMU design’ topic. These marginal results, however, do not affect the conclusions, but interestingly reflect how difficult it is, whether through natural language processing, or analytically, to disentangle between the different components of EMU.
# Topic 1 on ‘economic union’ encompasses discussion about the implications of the fixing of exchange rates in the monetary union, including the role of financial transfers of resources and regional policy. A comment from de Larosière encapsulated this well: “One has to explain that if one removes the exchange rate instrument, those countries with structural difficulties, where productivity is less, if they are deprived of the exchange rate instrument this will mean additional problems for them.” Turning to the content of economic union, Hoffmeyer insisted that “we have to specify exactly what the requirements in the monetary field and in the fiscal field are.” He then identifies five themes from the draft report: budgetary policy rules, welfare implications, tax harmonisation, wages, and financial transfer of resources. He insists that these “far-reaching” implications should be spelled out in plain to the heads of government: “it is for clarification, because politicians often only read summaries! You have to be very clear about what the demands are.” Delors suggested in the sixth meeting that “economic union be defined in terms of rules, like monetary union” and focuses on four main “pillars”: “rules governing competition,” “an environment that will ensure that the market will operate as well as possible,” a “Community policy that would concentrate on cooperation and regional development,” coordination of budgetary policy with constraints to be defined. Delors then explained the reasons behind his rather unambitious approach to ‘economic union’: “By virtue of the principle of subsidiarity we cannot do everything. In 1992 the Community budget will amount to 1.2% of gross GDP of member countries, perhaps at the end of the century we shall have 3% public expenditure. In other words, economic union will be less spectacular than monetary union for reasons of common sense.” Delors was even clearer in an exchange with Chalikias that was not transcribed. In insisting that rules should be strict, the Greek central banker asks for “more authority [to be given] to the European Commission for the implementation of macro-economic policy,” to which Delors, in closing the meeting, only replies: “it’s premature. As president of the Commission, it’s premature.” 
# Complementing the gLDA analysis of the transcripts with the listening to the tapes allows us to further refine the picture. While some remarks offered reflections on wage policy and regional policy, others raised the topic of economic union only to display some scepticism. In the fourth meeting, Delors thus warned that: “If the majority feel that economic union must precede monetary union then you should say so immediately because then you have to defend a different concept but this will lead to political failure because I already see what’s going to happen at the next summit.” Lamfalussy was on the same line: “We should be very careful of not giving the impression that all powers have to be transferred to a sort of European government. A, it’s not sellable politically, and b it is not realistic.” In spite of this Doyle regularly came back in the discussions to the topic of regional policy. This is partly due to his personal – and indeed the longstanding interest of the Irish government in – the topic, and also because he submitted a paper on regional policy to the Committee. In the tapes we can even listen to Doyle joking about his own obsessive interest in the topic: “I don’t want to give the impression that everything reminds me of regional policy like the Frenchman on sex, if I can say that!” Doyle’s regular interventions on regional policy also contribute to explain why he arrives fourth in the list of the longest speaking members in the Committee, as shown above.
# The issue of the difficulty of defining economic union, especially in contrast to the ease of defining what is monetary union, came up frequently. In an intervention during the same fourth meeting, Doyle thus explained: “monetary union, you can draw a circle around and write in the circle what it means, you cannot with economic union. (…) If you had enough political will and agreement in Europe you could set up a monetary union tomorrow, well, next year, let’s say, but it would not survive without having made a great deal of progress on what I’m calling for shorthand ‘economic union’.” To Pöhl who complained that “The emphasis in this skeleton is much too much on monetary integration instead of economic integration,” Lamfalussy replied: “if you want to begin with economic union, there is no a priori simple definition of what economic union is, there is no historical experience which is similar to any other historical experience in this respect, that is why, from a purely editorial point of view it is easier to begin with monetary union because the main characteristics of monetary union are known. Now I know that this may generate a wrong impression and one may wish to correct that impression but don’t ask the authors to start with economic union, because how are you gonna describe it?” And Delors concluded with what he implied was an unrealistic option: “Unless, as I indicated in my introduction, you say that this is a transfer of all powers of economic policy to the Community.”
# 
# In order to provide a further check on the gLDA topics, we compare and contrast the gLDA findings with the subjective summaries of the discussions that the two rapporteurs of the Committee drafted after each meeting except the last two. These documents were written for the purpose of having a clear summarised record of the main topics discussed, agreements reached, and issues to settle as the work of the Committee progressed. These summaries therefore represent a detailed and reliable track-record of what was discussed in the Committee. Table 5 below systematises the summaries provided by the rapporteurs into the meetings and topics. The table reports the themes, as presented and defined by the rapporteurs, by quoting directly from their summaries. This comparison confirms that the topics raised correspond indeed to the distribution of gLDA topics in the meetings presented in Table XXX.

# |Meeting|Themes|Sub-themes|Corresponding LDA topic|
# |:----|:----|:----|:----|
# |1|“Organisational aspects”|n/a|Topics 0 and 5|
# | |“Lessons drawn from the experience of the Werner Report”|“definition of EMU”|Topic 0|
# | | |“issue of parallelism”|Topic 0 and 1|
# | | |“nature of the intermediate steps”|Topics 0 and 4|
# | | |“nature of the monetary union (fixed but adjustable rates vs ‘fixed-fixed’)”|Topics 0 and 2|
# |2|“Characteristics and implications of an economic union consistent with a monetary union”|“problems arising in a situation of irrevocably fixed exchange rates”|Topic 1|
# | | |“policies needed to deal with these problems”|Topic 1|
# | | |“institutional arrangements required to formulate and execute such policies”|Topics 0 and 4|
# | |“a first exchange of views on an outline for the final report”|n/a|Topic 0|
# |3|“a first exchange of views on possible concrete steps towards economic and monetary union. The overwhelming part of the discussion centred on issues relating to institutional changes and, in particular, the question of whether a meaningful first concrete step could be made without or only with legal changes”|n/a|Topics 0 and 4|
# |4|“A general exchange of views developed on the skeleton. Considerable time was devoted to discussing whether the report should deal with the following question: if we have (a) the single market in 1992 do we also need (b) monetary union and (c) macro-economic policy coordination with financial transfers?”|n/a|Topic 0|
# | |“the rest of the meeting was devoted to a section-by-section discussion of Part II of the skeleton, i.e. on the part of the report dealing with the final stage of the economic and monetary union”|n/a|Topics 0 and 4|
# |5|“The meeting was devoted to the discussion of a new draft of Part III of the report, issues relating to the ECU”|“The general discussion of Part III centred mainly on a review of the two scenarios presented in Stage I, with a particular emphasis on the question of whether it would be desirable to describe two alternative scenarios in the final report” [one of the scenarios is the European Reserve Fund]; drafting suggestions; discussion on stages II and III”|Topics 0 and 3|
# | |“Issues related to the ECU: need for a single currency; parallel currency approach; link between the official and the private ECU; the ECU as a numéraire for private transactions and its role in promoting monetary union; the ECU as a monetary policy instrument”|n/a|Topics 0 and 3|
# | |“Organizational matters”|n/a|Topics 0 and 5|
# 
# 
# Table XXX: Summaries of the meetings provided by the rapporteurs.

# In[51]:


tlabel = pd.read_excel('./script/topics_manually_labelled.xlsx', 
                       sheet_name='guidedLDA',
                      )
tlabel


# In[52]:


d = d[d.columns.drop(list(d.filter(regex='Unnamed')))]
d = d[d['speaker_clean']!='?']


# In[53]:


d = pd.merge(d, tlabel, left_on='Dominant_Topic', right_on='topics', how='left')
d = d[d['text_clean'].notna()]


# In[54]:


d['len_'] = d.text_split.apply(lambda x: len(x.split()))
d['perc_'] = (d.len_ / d.len_.sum()) * 100


# In[55]:


d_topic_meeting = pd.pivot_table(d, 
                                  index='meeting',
                                  columns='manual_label_parent',
                                  values = 'intervention_id', 
                                  aggfunc='count',
                                 ).fillna(0)

d_topic_meeting_perc = pd.pivot_table(d, 
                                  index='meeting',
                                  columns='manual_label_parent',
                                  values = 'perc_', 
                                  aggfunc='sum',
                                 ).fillna(0).round(decimals=1)
d_topic_meeting_perc.loc['Totals'] = d_topic_meeting_perc.sum(numeric_only=True, axis=0)

d_topic_person_child = pd.pivot_table(d, 
                                  index='speaker_clean',
                                  columns='manual_label_child',
                                  values = 'intervention_id', 
                                  aggfunc='count',
                                 )

d_topic_person_parent = pd.pivot_table(d, 
                                  index='speaker_clean',
                                  columns='manual_label_parent',
                                  values = 'intervention_id', 
                                  aggfunc='count',
                                 )

d_topic_person_w = pd.pivot_table(d, 
                                  index='speaker_clean',
                                  columns='manual_label_parent',
                                  values = 'perc_',
                                  aggfunc= 'sum',
                                 ).fillna(0).round(decimals=1)


# In[56]:


d_topic_meeting_perc


# We can now compare and contrast the three elements: the topics identified through the gLDA analysis, the repartition of the gLDA topics by meeting, and the analysis of these meetings by the rapporteurs. The gLDA analysis therefore logically confirms that more time was spent in the discussions on the topics of substance related to EMU design and economic union (topics 0 and 1) rather than the four other topics. The overview provided in Table 5 highlights that topics are well distributed during individual meetings. Most importantly, as reflected in the summaries of the rapporteurs, topic 1 on economic union was extensively discussed during the second meeting devoted to the implications of locking exchange rate fluctuations (for which we have the second largest transcript with ca. 17000 words), and topic 1 emerges as the most discussed topic for the second meeting in the gLDA (Table XXX). Apart from a small predominance of Topic 0, other topics are fairly evenly distributed for the third meeting, but the transcript is relatively short (about 9000 words). The fourth meeting focused on discussing the skeleton report, which explains why two topics, EMU design and economic union stand out. The fifth meeting, for which we have the largest transcript (ca. 21000 words) focused on Chapter III of the Report and the proposal for a European Reserve Fund, and hence topic 0 dominates the discussion. The last two meetings dealt with the skeleton report, which explains that topics are again fairly evenly distributed, apart from the salience of topic 0 in the last meeting.

# In[57]:


d_topic_person_w
# Number of sentences for each topics per person


# Looking at the distribution of topics by speaker, we can see that some participants have more interventions attributed to the topic of economic union than any other. Lamfalussy and Delors are the two most interesting cases as they have substantial speaking time. Pöhl’s interventions are relatively evenly spread between topic 0 and 1 (10.7 vs 7.3). Interestingly, Doyle’s overall balance of interventions leans more towards EMU design than economic union, but only by a small margin (2.6 vs 1.8). Conversely, de Larosière’s interventions are very predominantly under the topic of EMU design. This is logical as the governor of the Banque de France developed and discussed his idea of a European Reserve Fund, which had more to do with the design of EMU (if not indeed monetary union alone) than issues related to economic union.
# One important topic is absent from the gLDA but important the design of EMU, namely, banking regulation and supervision. A close reading of the transcript and the summaries as well as a listening to the tapes contributes to explain this absence. The topic of banking regulation and supervision was tackled, but only very marginally, and thus is not substantial enough to form one of the five coherent topics identified by the gLDA. Banking regulation and supervision is not mentioned in the summaries of the meetings. The only intervention explicitly mentioning the topic is that of Duisenberg, ​​“I would also want to add in this list of guiding principles, (…) that this federation of central banks or central bank should be charged with banking supervision at the Community level.” The real substantial discussion happened elsewhere, in the Banking Supervisory Sub-Committee (BSSC) of the EEC Committee of Governors, that the Delors Report suggested creating, and that first met in March 1989. The whole subject of banking regulation and supervision was then delegated for discussion within the BSSC that gathered high-level representatives of the supervisory authorities of the EEC member states. The BSSC published a report on banking supervision in the European System of Central Banks (ESCB) and provided advice to the Committee of Governors, which then fed into the CoG’s contribution to the Intergovernmental Conference for the Maastricht Treaty. In that sense, the fact that LDA did not pick up banking regulation and supervision is revealing of the fact that it was, in a large part, overlooked and underestimated by the members of the Committee.
# 

# ## Papers submitted and papers not submitted to the Committee

# The fact that economic union was central to the discussions of the Delors Committee is even more interesting knowing that many papers that were related to this topic were in reality not circulated to the Committee members. The rapporteurs of the Delors Committee drew up detailed lists of the different papers at their disposal. They identified those papers that were distributed to the Committee members, and those papers that, in the end, did not get circulated. Figure XXX represents the authors of the papers and the theme to which their paper was attached.
# 
# 

# In[58]:


display(Image("./media/fig6.png"))

print('Figure XXX: authors and themes of the papers distributed to the Delors Committee (Source: ECB Archives and TPS papers, HAEU)')


# In[59]:


display(Image("./media/fig7.png"))

print('authors and themes of the papers not distributed to the Delors Committee (Source: ECB Archives and TPS papers, HAEU)')


# The papers distributed to the Delors Committee were for the most part authored by the Committee’s members, and revolved around the themes tackled by the Committee in equal manner: economic union, the ECU, monetary policy, and legal and institutional questions. By contrast, the papers not submitted to the Committee were predominantly penned by the Commission, and focused on various aspects related to the theme of economic union. This suggests a sidelining of the work carried out by the small group of Commission advisers on EMU to privilege the internal dynamics of the Committee.
# 

# ## Conclusions

# The broad theme of economic union was therefore extensively discussed during the meetings of the Delors Committee. The fact that the centrality of economic union in the discussion of the Committee was not translated into a more concrete outcome in the Report highlights both the difficulties of defining what ‘economic union’ exactly entails and of agreeing on specific measures to implement. The Delors Committee members noted well both aspects. They struggled to define economic union, and they observed – as Delors put it more bluntly – that there were limits to what powers could be transfers to the European, federal level. The inability to have concluded a more balanced agreement between the ‘E’ and the ‘M’ of EMU cannot be blamed on the Delors Committee members for having overlooked the topic. 
# What analysing in detail these transcripts and tapes allow us to observe is that the real ‘dead angle’ of the pre-Maastricht analysis of EMU was banking regulation and supervision. The consensus was that the framework of multiple directives on key subjects was enough. As much as for economic union, policymakers believed that there was no need to move supervisory tasks to the supranational level. Further to the centrality of economic union in the discussions of the Delors Committee, this article showed that the Delors Committee’s dynamics did not just revolve around the trio composed by the three most famous actors, that is, Delors, Pöhl and de Larosière. Not only were the three experts (Lamfalussy, Thygesen and Boyer) active participants in the debates, but Doyle, who advocated and spoke at length for regional policy. 
# Finally, we hope that this article provides a methodological example of the combination of multiple types of sources (tapes, transcriptions, handwritten notes) with digital tools and close-reading analysis of the primary sources. As the above analysis showed, digital tools provided a useful support in providing evidence of the presence of economic union-related discussions in the Delors Committee meeting, but these tools also had their limits, and need to be complemented with a qualitative analysis of the sources. Much talk about ‘economic union’ was actually not transcribed, and has been used in this article to restore the richness of the debate.
# 

# ## References

# <div class="cite2c-biblio"></div><div class="cite2c-biblio"></div><div class="cite2c-biblio"></div>
