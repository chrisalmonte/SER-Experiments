import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull, distance
from sklearn.cluster import KMeans

def get_representative_vad(df, emotion_col, vad_cols, target_points=2000):
    """
    Extracts representative VAD points for each emotion category.
    """
    representative_rows = []
    
    # Group the dataset by emotional category
    grouped = df.groupby(emotion_col)
    
    for emotion, group in grouped:
        print(f"Processing: {emotion} (Total samples: {len(group)})")        
        points = group[vad_cols].values
        original_indices = group.index.values 
        
        if len(points) <= target_points:
            representative_rows.extend(original_indices)
            continue
            
        # Convex Hull
        try:
            hull = ConvexHull(points)
            shell_indices_local = np.unique(hull.simplices)
        except Exception as e:
            # Fallback in case points are perfectly coplanar
            print(f"  Warning: Hull failed for {emotion}, falling back to random sample.")
            shell_indices_local = np.array([])
            
        # Map local indices back to the original dataframe's index
        shell_indices_global = original_indices[shell_indices_local]
        representative_rows.extend(shell_indices_global)
        
        # 2. Extract the Dense Core (K-Means)
        num_core_needed = target_points - len(shell_indices_local)
        
        if num_core_needed > 0:
            interior_mask = np.ones(len(points), dtype=bool)
            interior_mask[shell_indices_local] = False
            
            interior_points = points[interior_mask]
            interior_indices_global = original_indices[interior_mask]
            
            # --- THE FIX IS HERE ---
            # Find out how many strictly unique coordinates exist in the interior
            unique_interior_points = np.unique(interior_points, axis=0)
            max_possible_clusters = len(unique_interior_points)
            
            # K-Means cannot look for more clusters than there are unique points!
            # So, we take whichever number is smaller.
            actual_clusters_to_find = min(num_core_needed, max_possible_clusters)
            
            if actual_clusters_to_find > 0:
                kmeans = KMeans(n_clusters=actual_clusters_to_find, random_state=42, n_init="auto")
                kmeans.fit(interior_points)
                centroids = kmeans.cluster_centers_
                
                # "Snap" to nearest real data points
                dists = distance.cdist(centroids, interior_points, metric='euclidean')
                closest_local_indices = np.argmin(dists, axis=1)
                core_indices_global = np.unique(interior_indices_global[closest_local_indices])
                
                representative_rows.extend(core_indices_global)
            
            # Optional: Print a note so you know when the cap was hit
            if num_core_needed > max_possible_clusters:
                 print(f"    Note: Only {max_possible_clusters} unique core points available (requested {num_core_needed}).")

    # Return a new dataframe containing ONLY our representative points
    return df.loc[representative_rows].drop_duplicates()

def get_core_representatives(df, emotion_col, vad_cols, target_points=2000):
    """
    Extracts the most representative (dense core) VAD points for each emotion category,
    ignoring boundary extremes.
    """
    representative_rows = []
    
    # Group the dataset by emotional category
    grouped = df.groupby(emotion_col)
    
    for emotion, group in grouped:
        print(f"Processing: {emotion} (Total samples: {len(group)})")

        if len(group) <= target_points:
            print(f"  -> Only {len(group)} samples available, adding all to representatives.")
            representative_rows.extend(group.index.values)
            continue
        
        # Extract the 3D coordinates (Valence, Activation, Dominance)
        points = group[vad_cols].values
        original_indices = group.index.values 
        
        # 1. Safety Check: Count strictly unique physical coordinates
        unique_points = np.unique(points, axis=0)
        max_possible_clusters = len(unique_points)
        
        # K-Means cannot look for more clusters than unique points available
        actual_clusters = min(target_points, max_possible_clusters)
        
        if actual_clusters < target_points:
             print(f"  -> Note: Capping at {actual_clusters} unique core points (requested {target_points}).")

        if actual_clusters == 0:
            continue
            
        # 2. Extract the Prototypes (K-Means)
        # We fit K-Means on ALL points (not just unique ones) so the algorithm 
        # is correctly magnetically pulled toward the highest density areas.
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init="auto")
        kmeans.fit(points)
        centroids = kmeans.cluster_centers_
        
        # 3. "Snap" to Real Data
        # Calculate distance between every mathematical centroid and every real point
        dists = distance.cdist(centroids, points, metric='euclidean')
        
        # Find the index of the closest actual point for each centroid
        closest_local_indices = np.argmin(dists, axis=1)
        
        # Map back to the original dataframe indices and ensure we don't grab duplicate rows
        core_indices_global = np.unique(original_indices[closest_local_indices])
        
        representative_rows.extend(core_indices_global)

    # Return a new dataframe containing ONLY our representative points
    return df.loc[representative_rows].drop_duplicates()

# --- Example Usage ---
# core_df = get_core_representatives(
#     df=your_massive_dataframe, 
#     emotion_col='emotion', 
#     vad_cols=['valence', 'activation', 'dominance'], 
#     target_points=2000
# )

df = pd.read_csv(r"C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\custom\divided_labels_consensus_fs.csv")
df = df[df['Split_Set'] == 'Train']
df = df[~df['EmoClass'].isin(["X", "O"])]
undersampled = get_core_representatives(df, 'EmoClass', ['EmoAct', 'EmoVal', 'EmoDom'], target_points=3500)
undersampled.to_csv('divided_labels_emo_core_3500.csv', index=False)
