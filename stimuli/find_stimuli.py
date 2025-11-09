import pandas as pd
import glob



for file in glob.glob("*_summary.csv"):
    df = pd.read_csv(file)
    df["Target"] = "0"
    df.loc[df["POS_Tags"].str.contains(r"\bJJ\s*,\s*JJ\b", regex=True), "Target"] = "JJJ"
    df.loc[df["POS_Tags"].str.contains(r"\bNN\s*,\s*NN\b", regex=True), "Target"] = "NNN"
    df = df[df["Target"] != "0"]
    out_file = file.replace("_sentences_summary.csv", "_stimuli.csv")
    df.to_csv(out_file, index=False)
