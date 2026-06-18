# RAG and NLP Pipeline with UMAP

An intelligent data processing pipeline for analyzing customer feedback. This project integrates spam detection using RNNs, dimensionality reduction using UMAP, and Large Language Models to automatically categorize, evaluate, and search through customer messages.

## Key Features

### 1. Spam Detection with RNN
Before any text processing happens, the pipeline uses a pre-trained Recurrent Neural Network, carried over from a past project via `SpamRNN.py`. This model scores the probability that a message is spam, ensuring that the dataset is clean and that AI resources are not wasted on automated or irrelevant messages.

### 2. Semantic Clustering with UMAP
To group similar customer messages together organically:
- **Embeddings:** We use `SentenceTransformer` to convert every text message into a dense mathematical vector.
- **Dimensionality Reduction:** We use UMAP to compress these 384-dimensional vectors down into 2D coordinates.
- **Clustering:** K-Means automatically assigns the messages to relevant spatial categories.

### 3. Bulk AI Summary & More
We send batched data to the Gemini API using strict JSON parsing. The LLM acts as an expert analyst to read the message, look at the UMAP category, and assign:
- `Sentiment` (Positive/Negative/Neutral)
- `Urgency` (1 to 5 scale)
- `Summary` (A clean, one-sentence breakdown)

### 4. RAG
The pipeline features a native Retrieval-Augmented Generation implementation. When querying the dataset, the system:
1. Embeds the user's search query.
2. Uses cosine similarity to find the top 10 most relevant messages in the embedded database.
3. Passes those specific messages back to Gemini to synthesize an accurate, perfectly cited response directly in the console.
