import os
import shutil

def flatten_dataset():
    for split in ['train', 'valid', 'test']:
        if not os.path.exists(split):
            print(f"{split} directory not found. Ensure dataset is unzipped in the root.")
            continue
        for class_dir in os.listdir(split):
            class_path = os.path.join(split, class_dir)
            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    old_path = os.path.join(class_path, img)
                    clean_class = class_dir.replace('___', '_').replace(' ', '_')
                    new_name = f"{split}_{clean_class}_{img}"
                    new_path = new_name
                    if os.path.exists(new_path):
                        base, ext = os.path.splitext(new_name)
                        i = 1
                        while os.path.exists(f"{base}_{i}{ext}"):
                            i += 1
                        new_path = f"{base}_{i}{ext}"
                    shutil.move(old_path, new_path)
                    print(f"Moved {old_path} to {new_path}")
    # Remove empty directories
    for split in ['train', 'valid', 'test']:
        if os.path.exists(split) and not os.listdir(split):
            shutil.rmtree(split)
            print(f"Removed empty directory: {split}")

if __name__ == "__main__":
    flatten_dataset() 