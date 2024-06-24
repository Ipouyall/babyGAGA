from .preprocess import Cleanups
from osfclient.api import OSF  # pip install osfclient
import os
import zipfile


def download_from_osf(save_to, id="ad7qg", renew=True):
    def unzip_and_remove(zip_path):
        print(f'Unzipping {zip_path}...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(zip_path))
        os.remove(zip_path)
        print(f'Removed zip file {zip_path}')

    def download_files(storage, save_path):
        for file in storage.files:
            if "train" not in file.name and "test" not in file.name and "dev" not in file.name:
                continue
            file_path = os.path.join(save_path, file.name)
            if True:
                print(f'Downloading {file.name}...')
                with open(file_path, 'wb') as f:
                    file.write_to(f)
                # If the file is a zip file, unzip it and remove the zip file
                if file.name.endswith('.zip'):
                    unzip_and_remove(file_path)

    if os.path.exists(save_to) and renew:  # remove it
        os.system(f"rm -rf {save_to}")
    elif os.path.exists(save_to) and not renew:
        return

    osf = OSF()
    project = osf.project(id)
    storage = project.storage()

    os.makedirs(save_to, exist_ok=True)
    download_files(storage, save_to)


def clean(data_dir, proc_dir, rx_format):
    inst = Cleanups(
        data_dir=data_dir,
        proc_dir=proc_dir,
        rx_format=rx_format,
    )


if __name__ == "__main__":
    download_from_osf(
        save_to="./text_data/",
    )

    clean(
        data_dir="./text_data/train_10M",
        proc_dir="./processed/train_10M",
        rx_format="*train",
    )