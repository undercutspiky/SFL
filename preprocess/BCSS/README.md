## BCSS processing using code from WILDS package
The code here has been taken from the [WILDS package](https://github.com/p-lambda/wilds/tree/main). In particular from [this subdirectory](https://github.com/p-lambda/wilds/tree/main/dataset_preprocessing/camelyon17).


### Requirements
OpenCV must be installed before you install its corresponding Python wrapper listed in the requirements.txt file.

See installation instructions for OpenCV [here](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).

### Instructions

0. Download the data from https://github.com/PathologyDataScience/BCSS in your data directory (`--slide_root` argument you'll pass to files). You only need to download the directories called `masks` and `rgbs_colorNormalized`. Rename `rgbs_colorNormalized` to `images`. Your `<SLIDE_ROOT>` directory should contain 2 subdirectories called `masks` and `images`.


1. Run `python generate_all_patch_coords.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to generate a .csv of all potential patches. `<OUTPUT_ROOT>` is wherever you would like the patches to eventually be written. You can also skip this step and the next one if you simply use the `metadata.csv` provided here.


2. Then run `python generate_final_metadata.py --output_root <OUTPUT_ROOT>` to select a class-balanced set of patches. You can also skip this step and the previous one if you simply use the `metadata.csv` provided here.


3. Finally, run `python extract_final_patches_to_disk.py --slide_root <SLIDE_ROOT> --output_root <OUTPUT_ROOT>` to extract the chosen patches from the WSIs and write them to disk.
