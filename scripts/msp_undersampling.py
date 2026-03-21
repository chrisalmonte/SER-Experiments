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

df = pd.read_csv(r"C:\Datasets\MSP-PODCAST-Publish-2.0\Labels\custom\divided_labels_consensus_fs.csv")
df = df[df['Split_Set'] == 'Train']
undersampled = get_representative_vad(df, 'EmoClass', ['EmoAct', 'EmoVal', 'EmoDom'], target_points=3000)
undersampled.to_csv('msp_undersampled.csv', index=False)
