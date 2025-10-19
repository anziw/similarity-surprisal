import pandas as pd
import nltk
# nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# Provo
df = pd.read_csv("Provo_Corpus-Predictability_Norms.csv", encoding="unicode_escape")
unique_texts = df.drop_duplicates(subset=["Text_ID"])[["Text_ID", "Text"]]
unique_texts.to_csv("provo.csv", index=False)

# split Provo into sentences
df = pd.read_csv("provo.csv")

rows = []
sent_id = 1

for _, row in df.iterrows():
    text_id = row["Text_ID"]
    text = row["Text"]
    sentences = sent_tokenize(text)
    for sent in sentences:
        rows.append({
            "Text_ID": text_id,
            "Sent_ID": sent_id,
            "Sent": sent.strip()
        })
        sent_id += 1 

df_sentences = pd.DataFrame(rows)
df_sentences.to_csv("provo_sentences.csv", index=False)


# natural stories
df = pd.read_csv("words.tsv", sep="\t", header=None, names=["ID", "Text"])
df_whole = df[df["ID"].str.endswith(".whole")].copy()

df_whole["Text"] = df_whole["Text"].str.replace(" ", "", regex=False) # "it ." -> "it."
df_whole["Text_ID"] = df_whole["ID"].str.split(".", n=1).str[0]
combined = df_whole.groupby("Text_ID")["Text"].apply(" ".join).reset_index()
combined["Text_ID"] = combined["Text_ID"].astype(int)
combined = combined.sort_values("Text_ID")
combined.to_csv("naturalstories.csv", index=False)

# split natural stories into sentences
df = pd.read_csv("naturalstories.csv")

rows = []
sent_id = 1

for _, row in df.iterrows():
    text_id = row["Text_ID"]
    text = row["Text"]
    sentences = sent_tokenize(text)
    for sent in sentences:
        rows.append({
            "Text_ID": text_id,
            "Sent_ID": sent_id,
            "Sent": sent.strip()
        })
        sent_id += 1 

df_sentences = pd.DataFrame(rows)
df_sentences.to_csv("naturalstories_sentences.csv", index=False)
