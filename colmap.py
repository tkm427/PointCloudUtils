import pathlib

import pycolmap


def main():
    # SfM pipeline using pycolmap
    output_path = pathlib.Path("colmap")
    image_dir = pathlib.Path("data/output_frames/before3_conv")

    database_path = output_path / "database.db"

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Feature extraction
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir)

    # Feature matching
    print("Matching features...")
    pycolmap.match_exhaustive(database_path)

    # Incremental mapping (SfM)
    print("Running incremental mapping...")
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

    if maps:
        print(f"Successfully reconstructed {len(maps)} model(s)")
        maps[0].write(output_path)
        print(f"SfM reconstruction saved to {output_path}")
    else:
        print("Failed to reconstruct any models")
        return 1

    return 0


if __name__ == "__main__":
    main()
