import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

def main():
    if (len(sys.argv) < 2):
        print("Usage: python visualize.py <file>", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveliness', 'valence']
    sns.pairplot(df, hue = 'c', palette = 'hls', vars=features)
    plt.suptitle("K-Means Clustering: Spotify Audio Features")
    plt.show()
    # # After clustering
    # plt.figure()
    # df = pd.read_csv(sys.argv[1])
    # sns.scatterplot(x=df.danceability, y=df.energy, 
    #                 hue=df.c, 
    #                 palette=sns.color_palette("hls", n_colors=5))
    # plt.xlabel("danceability")
    # plt.ylabel("energy")
    # plt.title("Clustered: (y) vs (x)")

    # plt.show()

if __name__ == "__main__":
    main()