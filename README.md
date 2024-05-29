Project Organization: 

GSM_rsc_VGG_unity_project
--------------------------
- data: Berkeley Segmentation Database data
-- vseg: Sample responses from vseg same/diff task 

- seg-model-example: Example usage of FlexMM
- src: All underlying code and methods are here 

- venv: src/bin/activate venv to activate virtualenv for this project 

- out: (deprecated)
-- EXP150_NatImages_NeuroPixels: Segmentation maps for BSD images used in EXP150
-- EXPT: Stimulus design for EXPT_10_2023 (translation experiment)
-- Sessions_NaturalEnsemble_136: (empty)

EXPERIMENT DIRECTORIES will include fitted maps and Session data
----------------------
- EXP150_NatImages_NeuroPixels
-- includes fitted SegmentationMaps
-- .mat files with more information ie about BSD experiment ID
- Sessions_136_NaturalEnsemble_136
-- includes fitted SegmentationMaps (binned and unbinned)
-- includes Session responses (8534.mat)
- NN2015
-- 01-07.mat 
-- fitted maps (as individual .pkl files)
- NN2015
