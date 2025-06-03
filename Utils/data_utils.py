import os
import shutil
import random
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def restruct_dataset(base_path, output_dir='/content/organized_Dataset'):
    new_path = 'temp_dir'
    os.makedirs(new_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train', 'val']:
        split_path = os.path.join(base_path, split)

        for folder in os.listdir(split_path):
            if not os.path.isdir(os.path.join(split_path, folder)):
                continue

            plant_name, disease_name = folder.split("___")
            plant_name = plant_name.replace(",", "").replace(" ", "_")
            disease_name = disease_name.replace(",", "").replace(" ", "_")

            plant_folder_path = os.path.join(new_path, plant_name)
            os.makedirs(plant_folder_path, exist_ok=True)

            disease_folder_path = os.path.join(plant_folder_path, disease_name)
            os.makedirs(disease_folder_path, exist_ok=True)

            old_folder_path = os.path.join(split_path, folder)
            for image in os.listdir(old_folder_path):
                old_image_path = os.path.join(old_folder_path, image)
                if os.path.isfile(old_image_path):
                    shutil.move(old_image_path, os.path.join(disease_folder_path, image))

            shutil.rmtree(old_folder_path)
    create_dataset(new_path, output_dir, train_ratio=0.75, valid_ratio=0.2, test_ratio=0.05)     


def create_dataset(source_dir, output_dir, train_ratio=0.75, valid_ratio=0.2, test_ratio=0.05):

    os.makedirs(output_dir, exist_ok=True)

    for plant_name in os.listdir(source_dir):
        plant_dir = os.path.join(source_dir, plant_name)
        if not os.path.isdir(plant_dir):
            continue

        classes = [cls for cls in os.listdir(plant_dir) if os.path.isdir(os.path.join(plant_dir, cls))]

        if len(classes) < 2:
            print(f"Skipping {plant_name} because it has only one class.")
            continue

        train_dir = os.path.join(output_dir, plant_name, "train")
        valid_dir = os.path.join(output_dir, plant_name, "valid")
        test_dir = os.path.join(output_dir, plant_name, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for cls in classes:
            class_dir = os.path.join(plant_dir, cls)
            images = [img for img in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, img))]
            random.shuffle(images)

            num_images = len(images)
            num_train = int(num_images * train_ratio)
            num_valid = int(num_images * valid_ratio)

            train_images = images[:num_train]
            valid_images = images[num_train:num_train + num_valid]
            test_images = images[num_train + num_valid:]

            for img in train_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(train_dir, cls, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

            for img in valid_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(valid_dir, cls, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

            for img in test_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(test_dir, cls, img)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)


        print(f"Processed {plant_name}: {len(classes)} classes.")

    print("Dataset creation complete.")

def show_random_images(dataset_name, num_images=5):
    images = []
    dataset_path = os.path.join('/content/organized_Dataset',dataset_name)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)
                images.append((os.path.join(root, file), label))

    selected_images = random.sample(images, num_images)

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    if num_images == 1:
        axes = [axes]

    for ax, (img_path, label) in zip(axes, selected_images):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"{label}\n{img.size}", fontsize=8)

    plt.tight_layout()
    plt.show()

def load_data_generator(plant_name):
    dataset_path = os.path.join('/content/organized_Dataset', plant_name)

    train = os.path.join(dataset_path, 'train')
    val = os.path.join(dataset_path, 'valid')
    test = os.path.join(dataset_path, 'test')

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train,
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  target_size=(150,150))
    val_gen = val_datagen.flow_from_directory(val,
                                              batch_size=64,
                                              class_mode='categorical',
                                              target_size=(150,150))

    test_gen = val_datagen.flow_from_directory(test,
                                              batch_size=64,
                                              class_mode='categorical',
                                              target_size=(150,150))

    return train_gen, val_gen, test_gen
