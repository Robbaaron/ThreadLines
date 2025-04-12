# ThreadLines
NLP-powered network graph of men's fashion brands, styles, and Reddit sentiment.

**ThreadLines** is a data-driven exploration of men's fashion brands using Reddit discussions, natural language processing, clustering, and graph theory. By mining adjectives and style cues from r/malefashionadvice, the project builds visual and semantic connections between brands based on how people talk about them.

## Features

- 🔍 Scrapes Reddit posts mentioning fashion brands
- 🧠 Extracts adjectives using spaCy NLP
- 📊 Builds similarity matrices using cosine similarity
- 🌐 Clusters brands with K-Means
- 🕸️ Visualizes a style network with NetworkX
- ☁️ Generates word clouds and bar charts for top descriptors

## How It Works

1. Load brand data from CSV
2. Search Reddit posts with PRAW
3. Extract style & fit adjectives
4. Create brand vectors + cluster them
5. Build an interactive brand similarity graph

## Dependencies

- `pandas`, `numpy`, `praw`, `spacy`, `scikit-learn`
- `matplotlib`, `networkx`, `wordcloud`

## Run the Project

`python main.py`

### Make sure to add your Reddit API credentials in the script:

- `CLIENT_ID = "your_id"`
- `CLIENT_SECRET = "your_secret"`
- `USER_AGENT = "your_agent"`


Install with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm



