from pathlib import Path
import random
import shutil

# ============================================================
# SnapInspect AI - DeepPCB Dataset Preparation
# ============================================================

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# DeepPCB downloaded/cloned folder
DEEPPCB_ROOT = PROJECT_ROOT / "DeepPCB-master"

# Output dataset
OUTPUT_ROOT = PROJECT_ROOT / "dataset"

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducible split
RANDOM_SEED = 42

# DeepPCB class mapping
# DeepPCB:
# 1=open
# 2=short
# 3=mousebite
# 4=spur
# 5=copper/spurious_copper
# 6=pin-hole
CLASS_MAP = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 5,
    6: 4,
}

CLASS_NAMES = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "pin_hole",
    5: "spurious_copper",
}


def create_directories():
    """Create output dataset directories."""

    folders = [
        OUTPUT_ROOT / "images" / "train",
        OUTPUT_ROOT / "images" / "val",
        OUTPUT_ROOT / "images" / "test",

        OUTPUT_ROOT / "labels" / "train",
        OUTPUT_ROOT / "labels" / "val",
        OUTPUT_ROOT / "labels" / "test",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("Dataset directories created.")


def find_deep_pcb_data():
    """
    Locate DeepPCB PCBData directory.
    """

    possible_paths = [
        DEEPPCB_ROOT / "PCBData",
        PROJECT_ROOT / "DeepPCB-master" / "PCBData",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Found DeepPCB data: {path}")
            return path

    raise FileNotFoundError(
        "\nDeepPCB dataset was not found.\n"
        f"Expected location:\n{DEEPPCB_ROOT}\n\n"
        "Make sure DeepPCB-master is inside the project folder."
    )


def find_test_images(pcb_data):
    """
    Find all *_test.jpg images.

    DeepPCB contains:
        *_test.jpg -> defective/test PCB
        *_temp.jpg -> defect-free template

    We only train on *_test.jpg images.
    """

    images = list(pcb_data.rglob("*_test.jpg"))

    if not images:
        images = list(pcb_data.rglob("*_test.jpeg"))

    if not images:
        images = list(pcb_data.rglob("*_test.png"))

    print(f"Found {len(images)} test images.")

    return images


def find_annotation(image_path):
    """
    Find annotation file corresponding to an image.

    Example:

    00041000_test.jpg
    00041000.txt
    """

    stem = image_path.name.replace("_test.jpg", "")
    stem = stem.replace("_test.jpeg", "")
    stem = stem.replace("_test.png", "")

    # Search in the same directory first
    candidates = [
        image_path.parent / f"{stem}.txt",
        image_path.parent.parent / f"{stem}.txt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Last resort: search within PCBData
    matches = list(
        image_path.parents[0].rglob(
            f"{stem}.txt"
        )
    )

    if matches:
        return matches[0]

    return None


def convert_annotation(annotation_path, image_width, image_height):
    """
    Convert DeepPCB annotation:

        x1,y1,x2,y2,type

    into YOLO:

        class_id x_center y_center width height
    """

    yolo_lines = []

    with open(
        annotation_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Support comma or whitespace
            line = line.replace(",", " ")

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                x1 = float(parts[0])
                y1 = float(parts[1])
                x2 = float(parts[2])
                y2 = float(parts[3])
                original_class = int(parts[4])

            except ValueError:
                continue

            # Ignore background
            if original_class == 0:
                continue

            if original_class not in CLASS_MAP:
                print(
                    f"Warning: Unknown class "
                    f"{original_class} in {annotation_path}"
                )
                continue

            class_id = CLASS_MAP[original_class]

            # Clamp coordinates
            x1 = max(0, min(x1, image_width))
            x2 = max(0, min(x2, image_width))

            y1 = max(0, min(y1, image_height))
            y2 = max(0, min(y2, image_height))

            # Width and height
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width <= 0 or box_height <= 0:
                continue

            # YOLO normalized center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            x_center /= image_width
            y_center /= image_height

            box_width /= image_width
            box_height /= image_height

            yolo_lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{box_width:.6f} "
                f"{box_height:.6f}"
            )

    return yolo_lines


def get_image_size(image_path):
    """
    Get image dimensions without requiring OpenCV.
    """

    try:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.size

    except Exception as error:

        print(
            f"Could not read image {image_path}: {error}"
        )

        return None


def split_dataset(images):
    """
    Split images into train/validation/test.
    """

    random.seed(RANDOM_SEED)

    random.shuffle(images)

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_images = images[:train_end]

    val_images = images[
        train_end:val_end
    ]

    test_images = images[
        val_end:
    ]

    return (
        train_images,
        val_images,
        test_images
    )


def process_split(images, split_name):
    """
    Copy images and generate YOLO labels.
    """

    image_output = (
        OUTPUT_ROOT
        / "images"
        / split_name
    )

    label_output = (
        OUTPUT_ROOT
        / "labels"
        / split_name
    )

    successful = 0
    skipped = 0

    for index, image_path in enumerate(images, start=1):

        annotation_path = find_annotation(
            image_path
        )

        if annotation_path is None:

            print(
                f"Skipping: annotation not found "
                f"for {image_path.name}"
            )

            skipped += 1
            continue

        size = get_image_size(
            image_path
        )

        if size is None:

            skipped += 1
            continue

        image_width, image_height = size

        yolo_lines = convert_annotation(
            annotation_path,
            image_width,
            image_height
        )

        if not yolo_lines:

            print(
                f"Skipping: no valid defects "
                f"in {annotation_path.name}"
            )

            skipped += 1
            continue

        # Use unique filename
        output_image_name = (
            f"{index:06d}.jpg"
        )

        output_label_name = (
            f"{index:06d}.txt"
        )

        output_image = (
            image_output
            / output_image_name
        )

        output_label = (
            label_output
            / output_label_name
        )

        shutil.copy2(
            image_path,
            output_image
        )

        with open(
            output_label,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "\n".join(yolo_lines)
            )

        successful += 1

        if index % 100 == 0:

            print(
                f"{split_name}: "
                f"{index}/{len(images)} processed"
            )

    print(
        f"\n{split_name.upper()} completed:"
    )

    print(
        f"  Successful: {successful}"
    )

    print(
        f"  Skipped: {skipped}"
    )


def create_data_yaml():
    """
    Create YOLO data.yaml automatically.
    """

    yaml_path = (
        PROJECT_ROOT
        / "training"
        / "data.yaml"
    )

    content = """path: ../dataset

train: images/train
val: images/val
test: images/test

names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: pin_hole
  5: spurious_copper
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(content)

    print(
        f"\nCreated: {yaml_path}"
    )


def print_summary(
    train_images,
    val_images,
    test_images
):

    print("\n" + "=" * 60)

    print(
        "SnapInspect AI Dataset Preparation Complete"
    )

    print("=" * 60)

    print(
        f"Train images : {len(train_images)}"
    )

    print(
        f"Validation   : {len(val_images)}"
    )

    print(
        f"Test images  : {len(test_images)}"
    )

    print("\nClasses:")

    for class_id, name in CLASS_NAMES.items():

        print(
            f"  {class_id}: {name}"
        )

    print("\nDataset location:")

    print(
        OUTPUT_ROOT
    )

    print("\nNext step:")

    print(
        "python training/train.py"
    )

    print("=" * 60)


def main():

    print("=" * 60)

    print(
        "SnapInspect AI - DeepPCB Preparation"
    )

    print("=" * 60)

    # Step 1
    create_directories()

    # Step 2
    pcb_data = find_deep_pcb_data()

    # Step 3
    images = find_test_images(
        pcb_data
    )

    if not images:

        raise RuntimeError(
            "No DeepPCB test images found."
        )

    # Step 4
    train_images, val_images, test_images = (
        split_dataset(images)
    )

    print("\nDataset split:")

    print(
        f"Train: {len(train_images)}"
    )

    print(
        f"Validation: {len(val_images)}"
    )

    print(
        f"Test: {len(test_images)}"
    )

    # Step 5
    process_split(
        train_images,
        "train"
    )

    process_split(
        val_images,
        "val"
    )

    process_split(
        test_images,
        "test"
    )

    # Step 6
    create_data_yaml()

    # Step 7
    print_summary(
        train_images,
        val_images,
        test_images
    )


if __name__ == "__main__":
    main()