import argparse
import sys

from utils.wlasl_extractor import extract_wlasl_features


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe features from WLASL dataset")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        required=True,
        help="Path to the kaggle dataset (e.g., ~/.cache/kagglehub/.../versions/5)",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data",
        help="Directory to save the extracted features CSV",
    )
    parser.add_argument(
        "--max_words", 
        type=int, 
        default=25,
        help="Number of top words to extract (to keep processing/training time reasonable)",
    )
    
    args = parser.parse_args()
    
    csv_path = extract_wlasl_features(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_words=args.max_words
    )
    
    print(f"Extraction complete! Data saved to: {csv_path}")


if __name__ == "__main__":
    main()
