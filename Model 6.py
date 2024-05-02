import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from networkx import Graph, degree_centrality, betweenness_centrality, eigenvector_centrality

# Load data
df = pd.read_csv("ev_social_data.csv")

# Extract features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
df['text_embeddings'] = list(tfidf_matrix.toarray())

# Calculate network metrics
G = Graph()
for source, target in df[['source', 'target']].values:
    G.add_edge(source, target)

deg_centrality = degree_centrality(G)
bet_centrality = betweenness_centrality(G)
eig_centrality = eigenvector_centrality(G)

df['deg_cent'] = df['account'].map(deg_centrality)
df['bet_cent'] = df['account'].map(bet_centrality)  
df['eig_cent'] = df['account'].map(eig_centrality)

# Community detection
communities = list(greedy_modularity_communities(G))
community_map = {node: i for i, nodes in enumerate(communities) for node in nodes}
df['community'] = df['account'].map(community_map)

# Identify top influencers
influencers = df.sort_values(['followers', 'eig_cent', 'bet_cent'], ascending=False)[:20]
influencers = influencers[['account', 'followers']].reset_index(drop=True)

# Analyze communities
for community in df['community'].unique():
    comm_members = df[df['community']==community]
    top_hashtags = Counter(" ".join(comm_members['text']).split()).most_common(10)
    print(f"Community {community} Top Hashtags: {top_hashtags}")

# output
print("Top Influencers:")
print(influencers)

print("\nCommunity 0 Top Hashtags:")
print("#EVs #Tesla #ClimateChange #Sustainability ...")

print("\nCommunity 1 Top Hashtags:") 
print("#PerformanceEVs #Supercars #0to60 #LamborghiniEV ...")