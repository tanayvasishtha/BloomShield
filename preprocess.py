import os
import shutil

def flatten_dataset():
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/valid', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    splits = {
        'train': 'New Plant Diseases Dataset(Augmented)/train',
        'valid': 'New Plant Diseases Dataset(Augmented)/valid',
        'test': 'test'
    }
    for split, split_path in splits.items():
        if not os.path.exists(split_path):
            print(f"{split_path} directory not found.")
            continue
        for class_dir in os.listdir(split_path):
            class_path = os.path.join(split_path, class_dir)
            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    old_path = os.path.join(class_path, img)
                    clean_class = class_dir.replace('___', '_').replace(' ', '_')
                    new_name = f"{clean_class}_{img}"
                    new_path = os.path.join(f'data/{split}', new_name)
                    if os.path.exists(new_path):
                        base, ext = os.path.splitext(new_name)
                        i = 1
                        while os.path.exists(os.path.join(f'data/{split}', f"{base}_{i}{ext}")):
                            i += 1
                        new_path = os.path.join(f'data/{split}', f"{base}_{i}{ext}")
                    shutil.move(old_path, new_path)
                    print(f"Moved {old_path} to {new_path}")
    # Remove empty directories
    for split_path in splits.values():
        if os.path.exists(split_path) and not os.listdir(split_path):
            shutil.rmtree(split_path)
            print(f"Removed empty directory: {split_path}")
    # Remove the augmented folder if empty
    augmented_path = 'New Plant Diseases Dataset(Augmented)'
    if os.path.exists(augmented_path) and not os.listdir(augmented_path):
        shutil.rmtree(augmented_path)
        print(f"Removed empty directory: {augmented_path}")

if __name__ == "__main__":
    flatten_dataset() 