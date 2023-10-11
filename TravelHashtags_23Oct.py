# Importing necessary libraries
import pandas as pd
import json
import networkx as nx
from itertools import combinations
from collections import Counter
import leidenalg as la
import igraph as ig
from pyvis.network import Network
import requests  # Import the requests library
import matplotlib.cm as cm
import matplotlib

# Load the JSON data from the URL
url = 'https://raw.githubusercontent.com/BTBeast/Hashtag-Obsessions/main/Travel_IGPosts_20231010.json'
data = json.loads(requests.get(url).text)  # Load data using requests and convert it to JSON

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Handle missing values and extract all hashtags
df['hashtags'] = df['hashtags'].apply(lambda x: x if isinstance(x, list) else [])
all_hashtags = [hashtag for sublist in df['hashtags'] for hashtag in sublist if isinstance(sublist, list)]

# Getting the top 100 frequent hashtags
top_hashtags = [hashtag for hashtag, count in Counter(all_hashtags).most_common(1000)]

# Randomly sample a number of posts
sampled_df = df.sample(n=999, random_state=1)

# Extracting co-occurring top hashtags pairs from the sampled posts
sampled_df['top_hashtags_pairs'] = sampled_df['hashtags'].apply(
    lambda x: list(combinations(set(x) & set(top_hashtags), 2))
)

# Counting the frequency of each pair
cooccurring_top_hashtags = [
    pair for sublist in sampled_df['top_hashtags_pairs'] for pair in sublist
]
cooccurring_top_hashtags_counts = Counter(cooccurring_top_hashtags)

# Creating a network graph
G = nx.Graph()
for pair, frequency in cooccurring_top_hashtags_counts.items():
    hashtag1, hashtag2 = pair
    G.add_edge(hashtag1, hashtag2, weight=frequency)

# Convert NetworkX graph to igraph
ig_graph = ig.Graph.from_networkx(G)

# Applying the Leiden algorithm for community detection
partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

# Adding community information to the nodes in the graph
for node, community in zip(G.nodes(), partition.membership):
    G.nodes[node]['community'] = community

# Creating an interactive network graph using pyvis
nt = Network(notebook=True, height="750px", width="100%")

# Getting the top 50 hashtags by degree for display
top_50_hashtags = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:50]

# Adding labels to the nodes and setting node size based on degree and color based on community
cmap = cm.Set2  # Updated to use the attribute instead of the function

# Calculate the minimum and maximum degree in the graph
min_degree = min(dict(G.degree()).values())
max_degree = max(dict(G.degree()).values())

# Function to normalize the degree values
def normalize_degree(degree, min_degree, max_degree, new_min, new_max):
    return ((degree - min_degree) / (max_degree - min_degree)) * (new_max - new_min) + new_min

# Adding labels to the nodes and setting node size based on normalized degree and color based on community
for node in G.nodes():
    if node in top_50_hashtags:
        community = G.nodes[node]['community']
        degree = G.degree(node)
        normalized_degree = normalize_degree(degree, min_degree, max_degree, 5, 100)  # Normalize degree to the range of 5 to 100
        color = cmap(community / len(set(partition.membership)))  # Normalizing community number to [0,1] for color mapping
        color = list(color[:3])  # Convert RGB to list
        color.append(0.9)  # Append alpha channel, 0.9 for half transparency
        color = matplotlib.colors.rgb2hex(color, keep_alpha=True)  # Convert RGBA to hex
        nt.add_node(node, 
                    label=node, 
                    title=f'{node}\nConnections: {degree}', 
                    size=normalized_degree,  # Updated to use normalized degree
                    color=color,  # Updated color with transparency
                    font={"size": normalized_degree, "strokeWidth": 0, "color": "white"},  
                    shape='circle')  

# Adding edges
for edge in G.edges(data=True):
    if edge[0] in top_50_hashtags and edge[1] in top_50_hashtags:
        nt.add_edge(edge[0], edge[1], value=edge[2]['weight'])

# Adjusting the physics layout of the network to bring nodes of the same community closer together and reduce jittering
nt.set_options("""
{
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.02,
      "springLength": 100,
      "springConstant": 0.03,
      "damping": 0.8,
      "avoidOverlap": 0.7
    },
    "minVelocity": 0.1,
    "solver": "forceAtlas2Based",
    "timestep": 0.2,
    "adaptiveTimestep": true
  }
}
""".replace("'", '"'))  

# Save the interactive network graph as an HTML file
nt.save_graph("network_travel.html")
