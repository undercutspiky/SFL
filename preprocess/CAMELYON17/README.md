## CAMELYON17 processing using code from WILDS package
The code here has been taken from the [WILDS package](https://github.com/p-lambda/wilds/tree/main). In particular from [this subdirectory](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17).

You can either run the code here to download and process the data yourself or you can simply download the processed data from https://dataverse.no/dataset.xhtml?persistentId=doi:10.18710/NXPLFL
### Requirements
OpenCV and OpenSlide libraries must be installed before you install their corresponding Python wrappers listed in the requirements.txt file.

See installation instructions for OpenSlide [here](https://github.com/openslide/openslide-python) and for OpenCV [here](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).

### Instructions

0. Download the CAMELYON17's slide level segmentation masks from https://camelyon17.grand-challenge.org/Data/ in your data directory (`--slide_root` argument you'll pass to files). You'd need to find a file called `lesion_annotations.zip` on one of the servers where CAMELYON17 is hosted. There are multiple `lesion_annotations.zip` files so find the one for CAMELYON17 (instead of CAMELYON16). Unzip the file in the data directory. In the end you should have a directory called `lesion_annotations` containing all the XML files.


1. Download the CAMELYON17 data from https://camelyon17.grand-challenge.org/Data/ using `download_slides.py`. The data should have the you see in the `download_slides.py` file. You can also download the dataset manually. But note that the dataset is huge, so download only the 100 WSIs with lesion annotations, which by themselves are 600G. You can find out which WSIs have annotations by looking at the `lesion_annotations` folder. The patch extraction code expects `--slide_root` to contain the `lesion_annotations` and `training` directories with the `training` directory further having `center_<X>` folders with `X=[0, 1, 2, 3, 4]` for 5 centres. Each centre directory should have its corresponding `tif` files.


2. Run `python generate_all_patch_coords.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to generate a .csv of all potential patches as well as the tissue/tumor/normal masks for each WSI. `<OUTPUT_ROOT>` is wherever you would like the patches to eventually be written.


3. Then run `python generate_final_metadata.py --output_root <OUTPUT_ROOT>` to select a class-balanced set of patches and assign splits. I have assigned the split such that: a) both train and val splits have roughly 50% tumour and 50% non-tumour patches and more importantly b) all the patches from a patient are either in train or val split but not both as models can easily learn patient-specific patterns which do not generalise well.


4. Finally, run `python extract_final_patches_to_disk.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to extract the chosen patches from the WSIs and write them to disk.
