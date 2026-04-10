import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import threading
import time

def elapsed_time_thread():
    # Time on a seperate thread, so it doesn't interfere with the main thread

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        print(f"Elapsed Time: {minutes}m {seconds}s", end="\r")
        time.sleep(1)

def radar_chart(df, features):
    centroids = df.groupby('c')[features].mean()

    centroids = (centroids - centroids.min()) / (centroids.max() - centroids.min())

    # Number of features
    N = len(features)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette('hls', n_colors=len(centroids))

    for i, (cluster_id, row) in enumerate(centroids.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}', color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size = 11)
    ax.set_title("Cluster Centroids: Spotify Audio Features", size = 14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig("kmeans_clustering_radar.png")
    plt.show()
    print("Saved radar chart as kmeans_clustering_radar.png")

def main():


    # 1. Pairplot of the data, colored by cluster assignment
    if (len(sys.argv) < 2):
        print("Usage: python visualize.py <file>", file=sys.stderr)
        sys.exit(1)

    print("Would you like to sample ALL the data (WILL TAKE ~30 MINUTES)? If not,it will sample 10000 random rows from the CSV file) (y/n)")
    choice = input().lower()

    print("Reading CSV file...")
    df_full = pd.read_csv(sys.argv[1])

    if choice == 'y':
        print("Sampling all data...")
        df = df_full
    else:
        print("Sampling 10000 random rows from the CSV file...")
        df = df_full.sample(n=10000)

    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveliness', 'valence']
    print("Plotting Data....")
    timer = threading.Thread(target=elapsed_time_thread, daemon=True)
    timer.start()
    sns.pairplot(df, hue = 'c', palette = 'hls', vars=features)
    plt.suptitle("K-Means Clustering: Spotify Audio Features")
    

    ## Save as a PNG file

    if choice == 'y':
        filename = "kmeans_clustering_full.png"
    else:
        filename = "kmeans_clustering_sample.png"

    plt.savefig(filename)
    plt.show()
    print("Saved pairplot as " + filename)

    # 2. Radar Chart of cluster centroids
    print("Plotting Radar Chart of cluster centroids using FULL data...")
    radar_chart(df_full, features)



    

if __name__ == "__main__":
    main()

