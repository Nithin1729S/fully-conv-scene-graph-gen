import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

# Load only first N relationships from file
N = 100

with open("../relationships.json") as f:
    data = json.load(f)

G = nx.DiGraph()

count = 0
for image in data:
    for rel in image['relationships']:
        if count >= N:
            break
        try:
            subj = rel['subject']['names'][0].lower()
        except KeyError:
            subj = rel['subject']['name'].lower()
        try:
            obj = rel['object']['names'][0].lower()
        except KeyError:
            obj = rel['object']['name'].lower()
        
        predicate = rel['predicate'].lower()
        G.add_edge(subj, obj, relation=predicate)
        count += 1
    if count >= N:
        break

# =========================
# DEGREE DISTRIBUTION
# =========================
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())
total_degrees = dict(G.degree())

avg_in = np.mean(list(in_degrees.values()))
avg_out = np.mean(list(out_degrees.values()))
avg_total = np.mean(list(total_degrees.values()))

print("Average in-degree:", avg_in)
print("Average out-degree:", avg_out)
print("Average total degree:", avg_total)

# Boxplots
plt.figure(figsize=(8,6))
plt.boxplot(
    [list(in_degrees.values()), 
     list(out_degrees.values()), 
     list(total_degrees.values())],
    tick_labels=["In-degree", "Out-degree", "Total degree"]
)
plt.ylabel("Degree")
plt.title(f"Degree Distribution of VG Graph (first {N} relationships)")
plt.show()

# Log-log plot of degree distribution
degrees = list(total_degrees.values())
plt.figure(figsize=(6,5))
counts, bins = np.histogram(degrees, bins=20)
plt.scatter(bins[:-1], counts, alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Degree (log)")
plt.ylabel("Frequency (log)")
plt.title("Degree Distribution (log-log)")
plt.show()

# Identify outliers (top 1% nodes)
threshold = np.percentile(degrees, 99)
outliers = [n for n, d in total_degrees.items() if d >= threshold]
print("Outlier nodes:", outliers)

# =========================
# PATH LENGTHS
# =========================
if not nx.is_connected(G.to_undirected()):
    Gc = max(nx.connected_components(G.to_undirected()), key=len)
    G_sub = G.subgraph(Gc).copy()
else:
    G_sub = G

avg_path_len = nx.average_shortest_path_length(G_sub.to_undirected())
diameter = nx.diameter(G_sub.to_undirected())

print("Average path length:", avg_path_len)
print("Diameter:", diameter)
print("log(N):", math.log(len(G_sub)))

# find node pairs with diameter distance
diam_pairs = []
for u in G_sub.nodes():
    for v in G_sub.nodes():
        if u != v and nx.has_path(G_sub, u, v):
            if nx.shortest_path_length(G_sub, u, v) == diameter:
                diam_pairs.append((u, v))
print("Example node pairs with diameter:", diam_pairs[:5])

# =========================
# GEODESIC PATH LENGTH
# =========================
geo_avg = nx.average_shortest_path_length(G_sub.to_undirected())
print("Average geodesic path length (giant component):", geo_avg)

# =========================
# CLUSTERING COEFFICIENT
# =========================
clust_coeffs = nx.clustering(G_sub.to_undirected())
avg_cc = nx.average_clustering(G_sub.to_undirected())
print("Average clustering coefficient:", avg_cc)

deg_list = dict(G_sub.degree())
cc_values = [clust_coeffs[n] for n in G_sub.nodes()]
deg_values = [deg_list[n] for n in G_sub.nodes()]

plt.figure(figsize=(6,5))
plt.scatter(deg_values, cc_values, alpha=0.6)
plt.xscale('log')
plt.xlabel("Degree")
plt.ylabel("Clustering Coefficient")
plt.title("Clustering Coefficient vs Degree")
plt.show()

# =========================
# AVG CC by DEGREE GROUPS (FIXED)
# =========================
degrees_arr = np.array(list(deg_list.values()))
p25 = np.percentile(degrees_arr, 25)
p75 = np.percentile(degrees_arr, 75)

low_deg_nodes = [n for n, d in deg_list.items() if d <= p25]
high_deg_nodes = [n for n, d in deg_list.items() if d >= p75]

if low_deg_nodes:
    avg_cc_low = np.mean([clust_coeffs[n] for n in low_deg_nodes])
else:
    avg_cc_low = float('nan')

if high_deg_nodes:
    avg_cc_high = np.mean([clust_coeffs[n] for n in high_deg_nodes])
else:
    avg_cc_high = float('nan')

print("Avg CC (low-degree nodes):", avg_cc_low)
print("Avg CC (high-degree nodes):", avg_cc_high)
