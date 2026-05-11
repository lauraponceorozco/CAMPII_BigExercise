import os
import gzip
import shutil

root_dir = 'data_test' # Change this to your actual path if different

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.nii'):
            nii_path = os.path.join(root, file)
            gz_path = nii_path + '.gz'
            
            print(f"Compressing: {file}...")
            with open(nii_path, 'rb') as f_in:
                with gzip.open(gz_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Optional: Remove the original .nii file after successful compression
            os.remove(nii_path)

print("Done! All files are now .nii.gz")

"""
import torch

# This should return True if the driver and CUDA are set up correctly
print(f"Is CUDA available? {torch.cuda.is_available()}")

# This will tell you the name of your RTX card
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected by PyTorch.")
    """

