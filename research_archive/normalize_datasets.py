import pandas as pd
import numpy as np
from pathlib import Path

def normalize_row(row_features):
    # row_features is a 1D numpy array of length N*63
    normalized = np.zeros(len(row_features), dtype=np.float32)
    num_frames = len(row_features) // 63
    
    for i in range(num_frames):
        idx = i * 63
        frame_features = row_features[idx:idx+63]
        
        # If all zeros (padded frame), keep as zero
        if np.all(frame_features == 0):
            continue
            
        wrist_x = frame_features[0]
        wrist_y = frame_features[1]
        wrist_z = frame_features[2]
        
        max_dist = 0.0
        for j in range(21):
            lx = frame_features[j*3]
            ly = frame_features[j*3 + 1]
            dist = ((lx - wrist_x)**2 + (ly - wrist_y)**2)**0.5
            if dist > max_dist:
                max_dist = dist
        
        if max_dist == 0:
            max_dist = 1.0
            
        for j in range(21):
            normalized[idx + j*3] = (frame_features[j*3] - wrist_x) / max_dist
            normalized[idx + j*3 + 1] = (frame_features[j*3 + 1] - wrist_y) / max_dist
            normalized[idx + j*3 + 2] = (frame_features[j*3 + 2] - wrist_z) / max_dist
            
    return normalized

def process_csv(csv_path):
    print(f"Normalizing Data in {csv_path}...")
    df = pd.read_csv(csv_path, header=None)
    
    # If it has a header
    if df.iloc[0,0] == 'label':
        df = pd.read_csv(csv_path)
        labels = df.iloc[:, 0].values
        features = df.iloc[:, 1:].values
        norm_features = np.array([normalize_row(row) for row in features])
        norm_df = pd.DataFrame(norm_features, columns=df.columns[1:])
        norm_df.insert(0, 'label', labels)
        norm_df.to_csv(csv_path, index=False)
    else:
        # No header
        labels = df.iloc[:, 0].values
        features = df.iloc[:, 1:].values
        norm_features = np.array([normalize_row(row) for row in features])
        norm_df = pd.DataFrame(norm_features)
        norm_df.insert(0, 'label', labels)
        norm_df.to_csv(csv_path, index=False, header=False)
        
    print(f"Saved normalized data to {csv_path}")

def main():
    data_dir = Path("data")
    if (data_dir / "asl_data_auto.csv").exists():
        process_csv(data_dir / "asl_data_auto.csv")
    else:
        print("Missing asl_data_auto.csv")
        
    if (data_dir / "wlasl_data_naive.csv").exists():
        process_csv(data_dir / "wlasl_data_naive.csv")
    else:
        print("Missing wlasl_data_naive.csv")

if __name__ == "__main__":
    main()
