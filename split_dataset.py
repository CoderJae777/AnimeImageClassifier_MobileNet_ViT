import os
import shutil
import random

# CONFIG
SOURCE_DIR = "anime_dataset_raw"
TARGET_DIR = "anime_dataset"
SPLIT_RATIOS = (0.7, 0.2, 0.1)  # train, val, test
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def split_class_images(class_name):
    source_path = os.path.join(SOURCE_DIR, class_name)
    all_images = [f for f in os.listdir(source_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * SPLIT_RATIOS[0])
    val_end = train_end + int(total * SPLIT_RATIOS[1])

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split_name, file_list in splits.items():
        target_split_dir = os.path.join(TARGET_DIR, split_name, class_name)
        os.makedirs(target_split_dir, exist_ok=True)

        for img in file_list:
            shutil.copy2(os.path.join(source_path, img),
                         os.path.join(target_split_dir, img))

if __name__ == "__main__":
    for class_name in ["authentic", "generated"]:
        print(f"Splitting class: {class_name}")
        split_class_images(class_name)

    print("\n✅ Done! Images are split into train/val/test folders.")
