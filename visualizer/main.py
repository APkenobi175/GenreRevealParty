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

def main():
    if (len(sys.argv) < 2):
        print("Usage: python visualize.py <file>", file=sys.stderr)
        sys.exit(1)
    print("Reading CSV Output File....")
    df = pd.read_csv(sys.argv[1])
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveliness', 'valence']
    print("Plotting Data....")
    timer = threading.Thread(target=elapsed_time_thread, daemon=True)
    timer.start()
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

