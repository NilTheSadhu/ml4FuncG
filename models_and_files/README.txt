These folders contain a collection of files that pertains to my model training and testing as well as data pre-processing

The data processing/pre-processing and analysis has been compiled into a main file (the key parts of it at least) titled ATLAS_preprocessing_and_processing_truncated_relevant_parts.ipynb inside the scripts_tests_train_files folder 
Some early unet/ddpm testing/exploration code is also in that same folder

The later iterations of testing code is within the models and tasks file

Note that the files will need to be taken out of that folder to ensure the paths align with some of the files in my other folders (ex: my paired data loader/lazy loader inside the spatial_transcriptomics_dataset.py within the tasks folder). 

The main models and dependencies for them are split between utils tasks and models. Note that for utils. 

The framework for the main SRDiff model we were working towards is based on the following repo. We also took inspiration from the STDiff repo for things like how we formatted and made and independent noise scheduler class/file.

They are linked below: 

SRDIFF: https://github.com/LeiaLi/SRDiff/tree/main/tasks


STDIFF: https://github.com/fdu-wangfeilab/stDiff/tree/master/model