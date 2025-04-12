import csv
import os
import json
import pandas as pd
import numpy as np
import praw
import networkx as nx
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import spacy
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from wordcloud import WordCloud
from networkx.readwrite import json_graph

# Load spaCy language model.
nlp = spacy.load("en_core_web_sm")

# Global variables for Reddit API access (replace with your credentials).
CLIENT_ID = "CLIENT_ID_HERE"
CLIENT_SECRET = "CLIENT_SECRET_HERE"
USER_AGENT = "USER_AGENT_HERE"

# Global variables for relevant adjectives.
STYLES = {
    "edgy", "minimalist", "business casual", "streetwear", "athletic",
    "casual", "classic", "preppy", "modern", "sophisticated", "contemporary",
    "urban", "elegant", "refined", "trendy", "formal", "sporty", "luxury",
    "avant-garde", "eclectic", "rugged", "technical", "sustainable",
    "iconic", "bold", "youthful", "fast-fashion", "curated",
}
FIT = {
    "slim", "tailored", "regular", "relaxed", "athletic", "loose", "baggy",
    "skinny", "curvy", "straight", "cropped",
}

# Cache filename for processed data.
CACHE_FILE = "processed_data_cache.json"


class ClothingBrand:
    """
    A class representing a clothing brand with its attributes.
    """
    def __init__(self, name, styles="TBD", price_range="TBD", fit_types="TBD"):
        self.name = name
        self.styles = [s.strip() for s in styles.split(",")]
        self.price_range = price_range
        self.fit_types = [f.strip() for f in fit_types.split(",")]

    def __repr__(self):
        return self.name

    def display(self):
        """Display the brand's attributes."""
        print(f"Brand: {self.name}")
        print(f"Styles: {', '.join(self.styles)}")
        print(f"Price Range: {self.price_range}")
        print(f"Fit Types: {', '.join(self.fit_types)}")
        print("------------")
        
    def to_dict(self):
        """Convert the ClothingBrand object to a dictionary."""
        return {
            "name": self.name,
            "styles": self.styles,
            "price_range": self.price_range,
            "fit_types": self.fit_types
        }
    
    @staticmethod
    def from_dict(d):
        """Re-create a ClothingBrand object from a dictionary."""
        # Join the lists back to strings for the constructor.
        return ClothingBrand(
            d["name"],
            styles=",".join(d["styles"]),
            price_range=d["price_range"],
            fit_types=",".join(d["fit_types"])
        )


def load_brands_from_csv(filename="mens_fashion_data.csv"):
    """Load clothing brands from a CSV file and return a list of ClothingBrand objects."""
    df = pd.read_csv(filename)
    brands_list = []
    for _, row in df.iterrows():
        brands_list.append(
            ClothingBrand(row["brand"], row["styles"], row["price_range"], row["fit_types"])
        )
    return brands_list


def extract_relevant_adjectives(text, allowed_adjectives):
    """Process the text using spaCy and return a list of adjectives in the allowed set."""
    doc = nlp(text)
    extracted = [token.text.lower() for token in doc if token.pos_ == "ADJ"]
    return [adj for adj in extracted if adj in allowed_adjectives]


def aggregate_brand_adjectives(reddit_posts, allowed_adjectives):
    """
    Aggregates adjectives for each brand from Reddit posts.
    Returns a dictionary mapping ClothingBrand objects to a Counter of adjectives.
    """
    brand_adj_counter = {}
    for post in reddit_posts:
        brand = post["brand"]
        text = post["title"] + " " + post.get("selftext", "")
        adjectives = extract_relevant_adjectives(text, allowed_adjectives)
        if brand not in brand_adj_counter:
            brand_adj_counter[brand] = Counter()
        brand_adj_counter[brand].update(adjectives)
    return brand_adj_counter


def process_data():
    """Perform heavy processing: load brands, collect Reddit posts, NLP, clustering, etc."""
    print("Loading brands from CSV...")
    brands = load_brands_from_csv()

    print("Collecting Reddit posts (this may take a while)...")
    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    subreddit = reddit.subreddit("malefashionadvice")
    results = []
    for brand in brands:
        print(f"Searching posts for '{brand}'...")
        try:
            for submission in subreddit.search(brand.name, limit=50):
                results.append({
                    "brand": brand,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                })
        except Exception as e:
            print(f"Error for {brand}: {e}")

    allowed = STYLES.union(FIT)
    brand_adjs = aggregate_brand_adjectives(results, allowed)

    # Build vocabulary and create brand vectors.
    vocab = set()
    for counter in brand_adjs.values():
        vocab.update(counter.keys())
    vocab = sorted(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}

    brand_vectors = {}
    for brand, counter in brand_adjs.items():
        vector = np.zeros(len(vocab))
        for word, count in counter.items():
            vector[vocab_index[word]] = count
        brand_vectors[brand] = vector

    price_mapping = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}

    # For each brand, extend the adjective vector with its numeric price value.
    extended_brand_vectors = {}
    for brand, vector in brand_vectors.items():
        # Get the price value; default to 0 if not found.
        price_value = np.array([price_mapping.get(brand.price_range, 0)])
        # Concatenate the adjective vector and the price value.
        extended_vector = np.concatenate([vector, price_value])
        extended_brand_vectors[brand] = extended_vector

    brand_list = list(extended_brand_vectors.keys())
    vectors = np.array([extended_brand_vectors[brand] for brand in brand_list])
    similarity_matrix = cosine_similarity(vectors)

    # Clustering using KMeans.
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(vectors)
    clusters_dict = defaultdict(list)
    for i, brand in enumerate(brand_list):
        clusters_dict[clusters[i]].append(brand)

    # Build a network graph based on cosine similarity.
    G = nx.Graph()
    for i, brand1 in enumerate(brand_list):
        for j, brand2 in enumerate(brand_list):
            if i >= j:
                continue
            similarity = similarity_matrix[i, j]
            if similarity > 0.7:
                G.add_edge(brand1, brand2, weight=similarity)

    print("Data processing complete!\n")
    return {
        "brands": brands,
        "brand_adjs": brand_adjs,
        "brand_list": brand_list,
        "similarity_matrix": similarity_matrix,
        "clusters": clusters,
        "clusters_dict": clusters_dict,
        "G": G
    }


def save_cache_json(data, filename=CACHE_FILE):
    """Convert processed data to JSON-friendly structures and save to a file."""
    serializable_data = {}
    serializable_data["brands"] = [brand.to_dict() for brand in data["brands"]]
    serializable_data["brand_adjs"] = {
        brand.name: dict(data["brand_adjs"][brand]) for brand in data["brand_adjs"]
    }
    serializable_data["brand_list"] = [brand.name for brand in data["brand_list"]]
    serializable_data["similarity_matrix"] = data["similarity_matrix"].tolist()
    serializable_data["clusters"] = data["clusters"].tolist()
    serializable_data["clusters_dict"] = {
        str(k): [brand.name for brand in v] for k, v in data["clusters_dict"].items()
    }
    
    # Relabel graph nodes: convert ClothingBrand objects to their names.
    H = nx.relabel_nodes(data["G"], lambda x: x.name if isinstance(x, ClothingBrand) else x)
    # Explicitly set the edges keyword to avoid warnings.
    serializable_data["G"] = json_graph.node_link_data(H, edges="links")
    
    with open(filename, "w") as f:
        json.dump(serializable_data, f)
    print(f"Data cached to {filename}.")



def load_cache_json(filename=CACHE_FILE):
    """Load processed data from a JSON cache file and convert back into the proper objects."""
    if not os.path.exists(filename):
        return None
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Rebuild brands from dictionaries.
    brands = [ClothingBrand.from_dict(b) for b in data["brands"]]
    # Create a mapping from brand name to ClothingBrand object.
    brand_name_to_obj = {brand.name: brand for brand in brands}
    
    # Rebuild brand_adjs using the mapping.
    brand_adjs = {}
    for name, counter_dict in data["brand_adjs"].items():
        brand_adjs[brand_name_to_obj[name]] = Counter(counter_dict)
    
    # Rebuild brand_list.
    brand_list = [brand_name_to_obj[name] for name in data["brand_list"]]
    
    # Rebuild similarity_matrix.
    similarity_matrix = np.array(data["similarity_matrix"])
    
    # Rebuild clusters.
    clusters = np.array(data["clusters"])
    
    # Rebuild clusters_dict.
    clusters_dict = {}
    for k, v in data["clusters_dict"].items():
        clusters_dict[int(k)] = [brand_name_to_obj[name] for name in v]
    
    # Rebuild the NetworkX graph.
    G = json_graph.node_link_graph(data["G"])
    
    print("Loaded cached data from JSON!")
    return {
        "brands": brands,
        "brand_adjs": brand_adjs,
        "brand_list": brand_list,
        "similarity_matrix": similarity_matrix,
        "clusters": clusters,
        "clusters_dict": clusters_dict,
        "G": G
    }


def process_data_with_cache_json():
    """Attempt to load cached data from JSON; if not available, process and save it."""
    data = load_cache_json()
    if data is None:
        data = process_data()
        save_cache_json(data)
    return data


def display_network_graph(G):
    """Display the network graph using matplotlib."""
    plt.figure(figsize=(12, 12))
    pos = nx.kamada_kawai_layout(G, weight="weight", scale=1.0)
    nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10,
            font_color="black", font_weight="bold", edge_color="#E1783D", node_color="#78909C")
    plt.title("Brand Network Graph")
    plt.show()
    # other colors:  coral #F83C21, blue #43D7D1


def display_brand_graphs(selected_brand, brand_adjs):
    """Display a bar chart and word cloud for the selected brand's adjectives."""
    counter = brand_adjs.get(selected_brand, None)
    if counter and counter.most_common():
        # Bar chart for the top adjectives.
        most_common = counter.most_common(10)
        adjectives, counts = zip(*most_common)
        plt.figure(figsize=(8, 4))
        plt.bar(adjectives, counts, color="#A5B4A5")
        plt.xlabel("Adjective")
        plt.ylabel("Frequency")
        plt.title(f"Top Adjectives for {selected_brand.name}")
        plt.xticks(rotation=45)
        plt.show()

        # Word cloud visualization.
        wordcloud = WordCloud(width=800, height=400, background_color="white") \
            .generate_from_frequencies(counter)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {selected_brand.name}")
        plt.show()
    else:
        print(f"No adjectives data available for {selected_brand.name}.")


def brand_interaction_menu(processed_data):
    """Let the user pick a brand and choose an option: view adjectives, similar brands, or visualizations."""
    brands = processed_data["brands"]
    brand_list = processed_data["brand_list"]
    similarity_matrix = processed_data["similarity_matrix"]
    brand_adjs = processed_data["brand_adjs"]

    print("\nAvailable Brands:")
    for i, brand in enumerate(brands):
        print(f"{i + 1}. {brand.name}")
    choice = input("Select a brand by number or type its name: ").strip()

    selected_brand = None
    try:
        choice_int = int(choice)
        if 1 <= choice_int <= len(brands):
            selected_brand = brands[choice_int - 1]
    except ValueError:
        for brand in brands:
            if brand.name.lower() == choice.lower():
                selected_brand = brand
                break
    if selected_brand is None:
        print("Brand not found. Please try again.")
        return

    print(f"\nYou selected: {selected_brand.name}\n")
    print("Options:")
    print("1. Show adjectives most used to describe this brand")
    print("2. Show similar brands based on similarity")
    print("3. Show visualizations for this brand (bar chart and word cloud)")
    option = input("Choose an option (1, 2, or 3): ").strip()

    if option == "1":
        counter = brand_adjs.get(selected_brand, None)
        if counter and counter.most_common():
            print(f"\nTop adjectives for {selected_brand.name}:")
            for adj, count in counter.most_common(10):
                print(f"  {adj}: {count}")
        else:
            print(f"No adjectives found for {selected_brand.name}.")
    elif option == "2":
        try:
            idx = brand_list.index(selected_brand)
        except ValueError:
            print("Selected brand not in similarity matrix.")
            return
        similarities = similarity_matrix[idx]
        similar_indices = np.argsort(similarities)[::-1]
        similar_brands = []
        for i in similar_indices:
            if brand_list[i] == selected_brand:
                continue
            similar_brands.append((brand_list[i], similarities[i]))
            if len(similar_brands) >= 3:
                break
        if similar_brands:
            print(f"\nBrands similar to {selected_brand.name}:")
            for brand, sim in similar_brands:
                print(f"  {brand.name}: Similarity {sim:.2f}")
        else:
            print("No similar brands found.")
    elif option == "3":
        display_brand_graphs(selected_brand, brand_adjs)
    else:
        print("Invalid option. Please choose 1, 2, or 3.")


def main():
    print("Starting data processing. Please wait...\n")
    processed_data = process_data_with_cache_json()

    while True:
        print("\n--- Fashion Brand Analysis ---")
        print("1. Interact with a brand")
        print("2. Display network graph")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            brand_interaction_menu(processed_data)
        elif choice == "2":
            display_network_graph(processed_data["G"])
        elif choice == "3":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
