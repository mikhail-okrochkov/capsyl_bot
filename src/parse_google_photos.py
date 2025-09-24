import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import argparse
import sys


def parse_google_photos_metadata(json_file_path: str) -> Dict[str, Any]:
    """
    Parse a single Google Photos JSON metadata file.

    Args:
        json_file_path: Path to the JSON metadata file

    Returns:
        Dictionary with parsed metadata
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract photo name (title)
    photo_name = data.get("title", "")

    # Extract location data
    geo_data = data.get("geoData", {})
    if geo_data:
        latitude = geo_data.get("latitude")
        longitude = geo_data.get("longitude")
        altitude = geo_data.get("altitude")
        location = f"{latitude}, {longitude}" if latitude and longitude else None
    else:
        location = None
        latitude = None
        longitude = None
        altitude = None

    # Extract timestamp (using photoTakenTime if available, otherwise creationTime)
    photo_taken = data.get("photoTakenTime", {})
    creation_time = data.get("creationTime", {})

    if photo_taken.get("timestamp"):
        timestamp = int(photo_taken["timestamp"])
        formatted_time = photo_taken.get("formatted", "")
    elif creation_time.get("timestamp"):
        timestamp = int(creation_time["timestamp"])
        formatted_time = creation_time.get("formatted", "")
    else:
        timestamp = None
        formatted_time = ""

    # Convert timestamp to datetime
    datetime_obj = datetime.fromtimestamp(timestamp) if timestamp else None

    # Extract people names
    people = data.get("people", [])
    names = [person.get("name", "") for person in people if person.get("name")]
    names_str = ", ".join(names) if names else None

    return {
        "photo_name": photo_name,
        "location": location,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "timestamp": timestamp,
        "datetime": datetime_obj,
        "formatted_time": formatted_time,
        "names_list": names,
    }


def is_jpg_photo(filename: str) -> bool:
    """
    Check if a filename corresponds to a JPG photo (not MP4 or screenshot).

    Args:
        filename: The filename to check

    Returns:
        True if it's a JPG photo we want to process
    """
    filename_lower = filename.lower()

    # Skip MP4 files
    if filename_lower.endswith(".mp4"):
        return False

    # Skip screenshots
    if "screenshot" in filename_lower:
        return False

    # Only include JPG files
    if filename_lower.endswith((".jpg", ".jpeg")):
        return True

    return False


def parse_folder_metadata(folder_path: str, file_pattern: str = "*.json") -> pd.DataFrame:
    """
    Parse all JSON metadata files in a folder and create a DataFrame.
    Only processes JPG photos with valid location data (excludes MP4s, screenshots, and photos without GPS coordinates).

    Args:
        folder_path: Path to folder containing JSON metadata files
        file_pattern: Pattern to match JSON files (default: "*.json")

    Returns:
        DataFrame with parsed metadata
    """
    folder = Path(folder_path)
    json_files = list(folder.glob(file_pattern))

    if not json_files:
        print(f"No JSON files found in {folder_path} matching pattern {file_pattern}")
        return pd.DataFrame()

    parsed_data = []
    errors = []
    skipped_files = []

    for json_file in json_files:
        try:
            # Read the JSON to check the title/filename
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            title = data.get("title", "")

            # Skip if not a JPG photo we want to process
            if not is_jpg_photo(title):
                skipped_files.append(f"Skipped {title} (not a JPG or is MP4/screenshot)")
                continue

            metadata = parse_google_photos_metadata(str(json_file))

            # Skip if no valid location data (latitude and longitude must be non-zero)
            if (
                not metadata["latitude"]
                or not metadata["longitude"]
                or metadata["latitude"] == 0
                or metadata["longitude"] == 0
            ):
                skipped_files.append(f"Skipped {title} (no valid location data)")
                continue

            metadata["source_file"] = str(json_file)
            metadata["year_folder"] = folder.name  # Add the folder name
            parsed_data.append(metadata)

        except Exception as e:
            errors.append(f"Error parsing {json_file}: {str(e)}")

    if errors:
        print("Errors encountered:")
        for error in errors:
            print(f"  {error}")

    if skipped_files:
        print(f"Skipped {len(skipped_files)} files (non-JPG, MP4s, screenshots, or no location data)")

    df = pd.DataFrame(parsed_data)
    print(f"Successfully parsed {len(parsed_data)} JPG photos with location data from {folder_path}")

    return df


def parse_google_photos_archive(root_path: str) -> pd.DataFrame:
    """
    Parse Google Photos archive with 'Photos from 20XX' folder structure.
    Combines all JPG photo metadata with valid location data into a single DataFrame.

    Args:
        root_path: Path to the root directory containing 'Photos from 20XX' folders

    Returns:
        Combined DataFrame with all photo metadata (only photos with GPS coordinates)
    """
    root_folder = Path(root_path)

    # Find all 'Photos from 20XX' folders (2014-2025)
    photo_folders = []
    for year in range(2014, 2026):  # 2014 to 2025 inclusive
        folder_name = f"Photos from {year}"
        folder_path = root_folder / folder_name
        if folder_path.exists() and folder_path.is_dir():
            photo_folders.append(folder_path)

    if not photo_folders:
        print(f"No 'Photos from 20XX' folders found in {root_path}")
        return pd.DataFrame()

    print(f"Found {len(photo_folders)} photo folders to process:")
    for folder in photo_folders:
        print(f"  - {folder.name}")

    all_dataframes = []
    total_photos = 0

    # Process each folder
    for folder in photo_folders:
        print(f"\nProcessing {folder.name}...")
        df = parse_folder_metadata(str(folder))

        if not df.empty:
            all_dataframes.append(df)
            total_photos += len(df)

    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nCombined dataset: {len(combined_df)} total JPG photos with location data")

        # Sort by datetime
        if "datetime" in combined_df.columns:
            combined_df = combined_df.sort_values("datetime")

        return combined_df
    else:
        print("No data found in any folders")
        return pd.DataFrame()


def main():
    """
    Main function with command line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Parse Google Photos metadata from 'Photos from 20XX' folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python parse_google_photos.py /path/to/google/photos/root
  python parse_google_photos.py /path/to/google/photos/root --output my_photos.csv
  python parse_google_photos.py .  # Use current directory
        """,
    )

    parser.add_argument("root_path", help='Path to the root directory containing "Photos from 20XX" folders')

    parser.add_argument(
        "--output",
        "-o",
        default="google_photos_metadata.csv",
        help="Output CSV filename (default: google_photos_metadata.csv)",
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output, only show summary")

    args = parser.parse_args()

    # Validate root path exists
    root_path = Path(args.root_path)
    if not root_path.exists():
        print(f"Error: Root path '{args.root_path}' does not exist")
        sys.exit(1)

    if not root_path.is_dir():
        print(f"Error: Root path '{args.root_path}' is not a directory")
        sys.exit(1)

    if not args.quiet:
        print(f"Processing Google Photos from: {root_path.absolute()}")

    # Parse all JPG photos from Google Photos archive
    df = parse_google_photos_archive(str(root_path))

    if not df.empty:
        if not args.quiet:
            print(f"\nDataFrame shape: {df.shape}")
            print(f"\nColumns: {list(df.columns)}")
            print("\nFirst few rows:")
            print(df[["photo_name", "location", "datetime"]].head())

        # Save to CSV
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"\nDataFrame saved to {output_path.absolute()}")

        # Display some statistics
        print(f"\nSummary statistics:")
        print(f"- Total JPG photos with location data: {len(df)}")
        if "datetime" in df.columns and df["datetime"].notna().any():
            print(f"- Date range: {df['datetime'].min()} to {df['datetime'].max()}")

        # Photos by year folder
        if "year_folder" in df.columns:
            print(f"\nPhotos by year folder:")
            year_counts = df["year_folder"].value_counts().sort_index()
            for year_folder, count in year_counts.items():
                print(f"  - {year_folder}: {count} photos")
    else:
        print("No JPG photos with location data found to process")
        sys.exit(1)


if __name__ == "__main__":
    main()
