 

To adjust input settings modify filepaths in import_utils.py including: 

# `import_utils.py`
- EXP NAME (for organization purposes)
- Create new folder matching EXP NAME in parent folder of src 
- Images for the Session
- .mat data file for the session 

# `Session.py`
- Session class
- analysis methods that do not require segmentation are implemented here
    - ie. `Session.neuron_exclusion()`
- Load the .mat data file into a Session object 

# `SegmentationMap.py`
- SegmentationMap class
- calls functions in `src/segmentation` directory 
- Load the stimulus images from `import_utils.py`
- `run_{EXP_NAME}.py` files  run `SegmentationMap.fit_model()`

# `Analysis.py` 
- Analysis class 
- takes Session class and SegmentationMap class as input 
- calls functions in `src/analysis` directory
## `src/analysis` 
- `single_neuron_analysis.py`
- `pairwise_analysis.py`



