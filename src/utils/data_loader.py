import os
import random
import shutil
from tqdm import tqdm
import sys

def load_data(data_directory, train_split=0.6, val_split=0.2, test_split=0.2):
    print("------ PROCESS STARTED -------")

    # Check if splits sum to 1
    if train_split + val_split + test_split != 1:
        raise ValueError("Train, validation, and test splits must sum to 1.")

    # Get all unique file names without extension
    files = list(set([os.path.splitext(name)[0] for name in os.listdir(data_directory) if not name.startswith('.')]))

    print(f"--- This directory has a total number of {len(files)} unique files---")
    random.seed(42)
    random.shuffle(files)

    train_size = int(len(files) * train_split)
    val_size = int(len(files) * val_split)
    # The rest goes to test

    # Creating required directories
    train_path_img = os.path.join('../data', 'train', 'images')
    train_path_label = os.path.join('../data', 'train', 'labels')
    val_path_img = os.path.join('../data', 'val', 'images')
    val_path_label = os.path.join('../data', 'val', 'labels')
    test_path_img = os.path.join('../data', 'test', 'images')
    test_path_label = os.path.join('../data', 'test', 'labels')

    os.makedirs(train_path_img, exist_ok=True)
    os.makedirs(train_path_label, exist_ok=True)
    os.makedirs(val_path_img, exist_ok=True)
    os.makedirs(val_path_label, exist_ok=True)
    os.makedirs(test_path_img, exist_ok=True)
    os.makedirs(test_path_label, exist_ok=True)

    # Function to copy files safely
    def safe_copy(src_file, dest_file):
        try:
            shutil.copy2(src_file, dest_file)
        except IOError as e:
            print(f"Unable to copy file. {e}")
        except:
            print(f"Unexpected error: {sys.exc_info()}")

    # Copy files to train, validation, and test folders
    for i, filex in enumerate(tqdm(files)):
        if filex == 'classes':  # Assuming 'classes' is a special case file that should not be copied
            continue
        src_img = os.path.join(data_directory, filex + '.jpg')
        src_label = os.path.join(data_directory, filex + '.txt')

        if i < train_size:
            safe_copy(src_img, os.path.join(train_path_img, filex + '.jpg'))
            safe_copy(src_label, os.path.join(train_path_label, filex + '.txt'))
        elif i < train_size + val_size:
            safe_copy(src_img, os.path.join(val_path_img, filex + '.jpg'))
            safe_copy(src_label, os.path.join(val_path_label, filex + '.txt'))
        else:
            safe_copy(src_img, os.path.join(test_path_img, filex + '.jpg'))
            safe_copy(src_label, os.path.join(test_path_label, filex + '.txt'))

    print(f"------ Training data created with {train_split*100:.0f}% split {train_size} images -------")
    print(f"------ Validation data created with {val_split*100:.0f}% split {val_size} images -------")
    print(f"------ Test data created with {test_split*100:.0f}% split {len(files) - train_size - val_size} images -------")
    print("------ TASK COMPLETED -------")

    return (train_path_img, train_path_label), (val_path_img, val_path_label), (test_path_img, test_path_label)


# import os
# import random
# import shutil
# from tqdm import tqdm  # tqdm is used for showing a progress meter in loops
# import sys

# def load_data(data_directory, train_split=0.6, val_split=0.2, test_split=0.2):
#     # Print the start of the process
#     print("------ PROCESS STARTED -------")

#     # Ensure the sum of splits equals 1
#     if train_split + val_split + test_split != 1:
#         raise ValueError("Train, validation, and test splits must sum to 1.")

#     # Gather all unique filenames (without extensions) from the specified directory, ignoring hidden files
#     files = list(set([os.path.splitext(name)[0] for name in os.listdir(data_directory) if not name.startswith('.')]))

#     # Print the total number of unique files found
#     print(f"--- This directory has a total number of {len(files)} unique files---")

#     # Set a fixed seed for reproducibility and shuffle the files randomly
#     random.seed(42)
#     random.shuffle(files)

#     # Calculate the size of each dataset based on the provided splits
#     train_size = int(len(files) * train_split)
#     val_size = int(len(files) * val_split)
#     # The remainder of the files will be used for the test set

#     # Define paths for storing images and labels for each dataset
#     train_path_img = os.path.join('../data', 'train', 'images')
#     train_path_label = os.path.join('../data', 'train', 'labels')
#     val_path_img = os.path.join('../data', 'val', 'images')
#     val_path_label = os.path.join('../data', 'val', 'labels')
#     test_path_img = os.path.join('../data', 'test', 'images')
#     test_path_label = os.path.join('../data', 'test', 'labels')

#     # Create the necessary directories for storing datasets, if they don't already exist
#     os.makedirs(train_path_img, exist_ok=True)
#     os.makedirs(train_path_label, exist_ok=True)
#     os.makedirs(val_path_img, exist_ok=True)
#     os.makedirs(val_path_label, exist_ok=True)
#     os.makedirs(test_path_img, exist_ok=True)
#     os.makedirs(test_path_label, exist_ok=True)

#     # Define a function to copy files, handling any errors that occur
#     def safe_copy(src_file, dest_file):
#         try:
#             shutil.copy2(src_file, dest_file)  # Uses copy2 to also copy metadata
#         except IOError as e:  # Handle file I/O errors
#             print(f"Unable to copy file. {e}")
#         except:  # Handle other unexpected errors
#             print(f"Unexpected error: {sys.exc_info()}")

#     # Copy files to the appropriate dataset directories
#     for i, filex in enumerate(tqdm(files)):  # tqdm provides a progress bar
#         if filex == 'classes':  # Skip copying a file named 'classes', if present
#             continue
#         src_img = os.path.join(data_directory, filex + '.jpg')
#         src_label = os.path.join(data_directory, filex + '.txt')

#         # Determine the destination based on the current index and specified splits
#         if i < train_size:
#             safe_copy(src_img, os.path.join(train_path_img, filex + '.jpg'))
#             safe_copy(src_label, os.path.join(train_path_label, filex + '.txt'))
#         elif i < train_size + val_size:
#             safe_copy(src_img, os.path.join(val_path_img, filex + '.jpg'))
#             safe_copy(src_label, os.path.join(val_path_label, filex + '.txt'))
#         else:
#             safe_copy(src_img, os.path.join(test_path_img, filex + '.jpg'))
#             safe_copy(src_label, os.path.join(test_path_label, filex + '.txt'))

#     # Print summary information about the created datasets
#     print(f"------ Training data created with {train_split*100:.0f}% split {train_size} images -------")
#     print(f"------ Validation data created with {val_split*100:.0f}% split {val_size} images -------")
#     print(f"------ Test data created with {test_split*100:.0f}% split {len(files) - train_size - val_size} images -------")
#     print("------ TASK COMPLETED -------")

#     # Return paths to the image and label directories for each dataset
#     return (train_path_img, train_path_label), (val_path_img, val_path_label), (test_path_img, test_path_label)
