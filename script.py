import tensorflow as tf
import pandas as pd
from dotenv import load_dotenv
from litellm import completion
import os
from sentence_transformers import SentenceTransformer
import plotly.express as px
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
import re



load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
df = pd.read_excel("input_file.xlsx")

    
    
print("\n-----PREDICTING SPAM------")
spam_model = tf.keras.models.load_model("spam_model.keras")
if 'Message' in df.columns:
    messages = df['Message'].astype(str)
    x_pred = tf.convert_to_tensor(messages.tolist(), dtype=tf.string)
    y_pred_probs = spam_model.predict(x_pred)
    df['prob_of_spam'] = y_pred_probs
    df.sort_values(by='prob_of_spam', ascending=True, inplace=True)
    


print("\n-----EMBEDDING AND LABELING TEXT-------")
def text_embedder(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, batch_size=5, show_progress_bar=True)
    return embeddings.tolist()

def umap_display(embeddings, hover_text, category_labels, color_label='Category'):
    embeddings = np.array(embeddings)
    n_clusters = len(category_labels)
    
    umap_reducer = umap.UMAP(
        n_components=2,
        n_neighbors=3,
        min_dist=0.1,
        random_state=42,
    )

    umap_embeddings = umap_reducer.fit_transform(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(umap_embeddings)
    
    mapped_categories = [category_labels[c] for c in clusters]

    umap_df = pd.DataFrame({
        "index": range(0, len(embeddings)),
        "UMAP1": umap_embeddings[:, 0],
        "UMAP2": umap_embeddings[:, 1],
        "label": hover_text,
        color_label: mapped_categories, 
    })

    fig = px.scatter(
        umap_df,
        x="UMAP1",
        y="UMAP2",
        color=color_label,
        hover_name="label",
        title=f"UMAP projection of text message embeddings (colored by {color_label})",
        labels={
            "UMAP1": "UMAP dimension 1",
            "UMAP2": "UMAP dimension 2",
        },
        hover_data=["index"],
    )
    
    fig.write_html("umap_plot.html")
    fig.show()
    return mapped_categories
    
    
category_names = ["Delivery Problem", "Foreign Comments", "Pots feedback",
                  "Shipping Questions and Feedback", "Delivery Problem", "Website Bugs", "Monstera Problems", 
                  "Packaging Positive Feedback", "Spam", "Positive Feedback", "Other Questions", "Plants Questions"]

embeddings = text_embedder(df["Message"].tolist())
np.save("embeddings.npy", np.array(embeddings))
print("Embeddings saved to embeddings.npy")

messages = df["Message"].tolist()
mapped_categories = umap_display(embeddings, messages, category_names, color_label='Category')

df['Probable_Category'] = mapped_categories
df.to_excel("output_file.xlsx", index=False)




print("\n-----AI ANALYSIS------")
MODEL_NAME = "gemini/gemini-3.1-flash-lite"
def chatWithLLM(prompt):
    try:
        response = completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

BATCH_SIZE = 50
all_results = []
print(f"Processing {len(df)} rows in batches of {BATCH_SIZE}")

for i in range(0, len(df), BATCH_SIZE):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    dataset_str = batch_df[['Message', 'prob_of_spam', 'Probable_Category']].to_csv(index=False)
    
    bulk_prompt = f"""
Here is a dataset of {len(batch_df)} customer messages in CSV format. Each row includes the 'Message', the AI-predicted 'prob_of_spam' (which is not perfect), and an automatically assigned 'Probable_Category' (also not perfect).
Some of the Messages may be provided in language different than English - translate them and create summary for them in English.

Please analyze EVERY SINGLE ROW in this dataset and return a JSON array of objects. For each message, provide the following fields, taking into account its given spam probability and given category to inform your decisions:
1. "Category": The most accurate category label (you can use my provided one or adjust it if mine is inaccurate).
2. "Sentiment": Either "Positive", "Negative", or "Neutral".
3. "Urgency": A priority score from 1 to 5 based on the issue's severity.
4. "Summary": A one-sentence summary of the user's comment.

Output ONLY valid JSON without any markdown formatting blocks. Do not wrap the JSON in ```json ```. Just raw JSON. The array must contain exactly {len(batch_df)} items in the exact same order as the provided dataset.

Dataset:
{dataset_str}
"""
    print(f"Sending batch {i//BATCH_SIZE + 1} ({len(batch_df)} messages) to Gemini...")
    ai_analysis = chatWithLLM(bulk_prompt)
    
    try:
        match = re.search(r'\[.*\]', ai_analysis, re.DOTALL)
        if match:
            results = json.loads(match.group(0))
        else:
            clean_json = ai_analysis.replace("```json", "").replace("```", "").strip()
            results = json.loads(clean_json)
            
        if len(results) == len(batch_df):
            all_results.extend(results)
        else:
            all_results.extend([{"Category": "Error", "Sentiment": "Neutral", "Urgency": "1", "Summary": "Parsing error"}] * len(batch_df))
    except Exception as e:
        all_results.extend([{"Category": "Error", "Sentiment": "Neutral", "Urgency": "1", "Summary": "Parsing error"}] * len(batch_df))




print("\n----SAVING TO EXCEL -----")
if len(all_results) == len(df):
    df["AI_Final_Category"] = [r.get("Category", "") for r in all_results]
    df["Sentiment"] = [r.get("Sentiment", "") for r in all_results]
    df["Urgency"] = [r.get("Urgency", "1") for r in all_results]
    df["Summary"] = [r.get("Summary", "") for r in all_results]
    
    df.to_excel("output_file.xlsx", index=False)






print("\n----- RAG: SEMANTIC SEARCH & GENERATION ------")
search_query = "What are people saying about plant packaging?"
print(f"User Query: '{search_query}'")

query_embedding = text_embedder([search_query])[0]
similarities = cosine_similarity([query_embedding], embeddings)[0]

top_10_indices = np.argsort(similarities)[-10:][::-1]
top_messages = [messages[i] for i in top_10_indices]

print("\nTop 10 matching messages found in database:")
for i, msg in enumerate(top_messages, 1):
    print(f"{i}. {msg}")

rag_prompt = f"""
You are a helpful customer insights AI. 
A team member has asked: "{search_query}"

Based ONLY on the following top 10 relevant customer messages retrieved from our database, answer their question:
1. "{top_messages[0]}"
2. "{top_messages[1]}"
3. "{top_messages[2]}"
4. "{top_messages[3]}"
5. "{top_messages[4]}"
6. "{top_messages[5]}"
7. "{top_messages[6]}"
8. "{top_messages[7]}"
9. "{top_messages[8]}"
10. "{top_messages[9]}"
"""

print("\nGenerating RAG response")
rag_response = chatWithLLM(rag_prompt)
print("\n--- RAG Response ---")
print(rag_response)

