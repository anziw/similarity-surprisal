import pandas as pd

df_RT = pd.read_csv("processed_RTs.tsv", sep="\t")
df_sent = pd.read_csv("naturalstories_sentences.csv")
df_RT["Sent_ID"] = 0

start = 0
end = 0

for _, row in df_sent.iterrows():
    sent = row["Sent"]
    sent_id = row["Sent_ID"]
    sent_content = sent.split()
    n = df_RT.iloc[start]['nItem']
    end += len(sent_content) * n
    df_RT.loc[start:end-1, "Sent_ID"] = sent_id
    start = end

df_RT.to_csv("naturalstories_RTs_sentid.csv", index=False)