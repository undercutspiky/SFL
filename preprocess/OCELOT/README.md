## OCELOT processing using code from WILDS package
The code here has been taken from the [WILDS package](https://github.com/p-lambda/wilds/tree/main). In particular from [this subdirectory](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17).


### Requirements
OpenCV and OpenSlide libraries must be installed before you install their corresponding Python wrappers listed in the requirements.txt file.

See installation instructions for OpenSlide [here](https://github.com/openslide/openslide-python) and for OpenCV [here](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).

### Instructions

Since the data has images and its annotations available at a resolution lower than 40x, we need to do the following additional steps:


0. Download the data from https://ocelot2023.grand-challenge.org/datasets/ in your data directory. For this project I used the version 1.0.1 (the latest version available at the time).
1. Download the original WSIs from TCGA so as to have images at 40x. This requires extracting information from OCELOT's `metadata.json` file and then creating a corresponding gdc manifest file so download the images. I have done all of that. The manifest file is present in this directory as `gdc_manifest.txt`. Use the GDC data transfer tool called `gdc-client` to download the WSIs. The tool and its documentation are available [here](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/). The WSIs (the `svs` files from TCGA) should be stored in a directory called `wsis` in the `<SLIDE_ROOT>` directory.
2. Run `python extract_initial_patches.py` after modifying the paths in `'__main__'` to extract the images (crops from the downloaded WSIs) that OCELOT has annotations for. The code expects `metadata.json` (downloaded from OCELOT's website but also provided here) to be present in the same directory as the code. Modify the path to wherever your `metadata.json` is.


After having done all the steps above, we're now ready to do the same stuff we did for CAMELYON17 and BCSS:


1. Run `python generate_all_patch_coords.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to generate a .csv of all potential patches. `<OUTPUT_ROOT>` is wherever you would like the patches to eventually be written. You can also skip this step and the next one if you simply use the `metadata.csv` provided here.

2. Then run `python generate_final_metadata.py --output_root <OUTPUT_ROOT>` to select a class-balanced set of patches. You can also skip this step and the previous one if you simply use the `metadata.csv` provided here.

3. Finally, run `python extract_final_patches_to_disk.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to extract the chosen patches from the WSIs and write them to disk.
