import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline

model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="first")

input_files = [
    "provo/provo_sentences.csv",
    "naturalstories/naturalstories_sentences.csv"
]

for file in input_files:
    df = pd.read_csv(file)
    rows = []
    summary_rows = []

    for _, row in df.iterrows():
        text_id = row["Text_ID"]
        sent_id = row["Sent_ID"]
        sent = row["Sent"]

        outputs = pipeline(sent)

        for i, out in enumerate(outputs, start=1):
            rows.append({
                "Text_ID": text_id,
                "Sent_ID": sent_id,
                "Word_Number": i,
                "Word": out["word"],
                "POS_Tag": out["entity_group"]
            })

        pos_seq = " ".join(out["entity_group"] for out in outputs)
        summary_rows.append({
            "Text_ID": text_id,
            "Sent_ID": sent_id,
            "Sent": sent,
            "POS_Tags": pos_seq
        })

    tagged_file = file.replace(".csv", "_tagged.csv")
    pd.DataFrame(rows).to_csv(tagged_file, index=False)

    summary_file = file.replace(".csv", "_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
