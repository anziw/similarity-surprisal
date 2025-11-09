import pandas as pd

files = ["naturalstories/naturalstories_sentences_summary.csv", "provo/provo_sentences_summary.csv"]

for file in files:
    df = pd.read_csv(file)
    df["Target"] = "0"
    df.loc[df["POS_Tags"].str.contains(r"\bJJ\s*,\s*JJ\b", regex=True), "Target"] = "JJJ"
    df.loc[df["POS_Tags"].str.contains(r"\bNN\s*,\s*NN\b", regex=True), "Target"] = "NNN"
    out_file = file.replace("_sentences_summary.csv", "_labeled.csv")
    df.to_csv(out_file, index=False)
