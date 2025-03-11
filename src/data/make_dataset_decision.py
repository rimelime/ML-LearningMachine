import kagglehub
import pandas as pd
import os
import shutil

# Download latest version
dataset_path = kagglehub.dataset_download("ramjasmaurya/medias-cost-prediction-in-foodmart")
files = os.listdir(dataset_path)

fileName = files[0]
srcPath = os.path.join(dataset_path, fileName)
rawFolder = os.path.join("data", "raw")

destinationPath = os.path.join(rawFolder, fileName)
shutil.move(srcPath, destinationPath)

print(f"Moved {srcPath} to {destinationPath}")