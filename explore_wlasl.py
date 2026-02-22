import kagglehub
import os

print("Downloading WLASL dataset...")
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
print("Path to dataset files:", path)

print("\nFiles in root:")
for f in os.listdir(path):
    print(f" - {f}")

# Check any subdirectories
for root, dirs, files in os.walk(path):
    if root != path:
        print(f"\nIn {os.path.relpath(root, path)}:")
        print(f"  Dirs: {len(dirs)}")
        print(f"  Files: {len(files)}")
        if files:
            print(f"  Sample files: {files[:5]}")
        break  # just check one level deep for large dirs
