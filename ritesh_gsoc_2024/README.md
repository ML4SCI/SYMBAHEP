# S-KANformer: Enhancing Transformers for Symbolic Calculations in High Energy Physics

This repository contains the source code for the above-mentioned title under the [SYMBA project](https://ml4sci.org/gsoc/2024/proposal_SYMBA1.html). This project is also a part of the [Google Summer of Code 2024](https://summerofcode.withgoogle.com/programs/2024/projects/0Oa841IT).
## Directory Structure

```
.
├── Data_generation/
│   ├── ew_general_diag.cpp
│   ├── qcd_general_diag.cpp
│   ├── qed_eea_diag.cpp
│   └── qed_general_diag.cpp
├── Data_preprocess/
│   ├── augment_data.py
│   ├── sequence_count.py
│   └── splitter.py
├── Models/
│   ├── SineKAN/
│   └── vanilla/
├── .gitignore
└── README.md
```

### Explanation of Directories and Files

1. **Data_generation/**  
   This folder contains the source files used for generating symbolic physics data related to electroweak (EW), quantum chromodynamics (QCD), and quantum electrodynamics (QED) processes using [MARTY](https://marty.in2p3.fr/).
   - `ew_general_diag.cpp`: Generates diagrams related to electroweak interactions.
   - `qcd_general_diag.cpp`: Handles data generation for QCD general diagrams.
   - `qed_eea_diag.cpp`: Generates data related to QED electron-positron annihilation (EEA).
   - `qed_general_diag.cpp`: Handles general QED diagram generation.
   
2. **Data_preprocess/**  
   This folder contains scripts for preprocessing the generated data before tokenization.
   - `normalize_data.py`: Replaces & normalizes the MARTY-generated indices in the sequences with custom numbered indices to limit the total number of tokens in vocabulary and bring uniformity.    
   - `sequence_count.py`: Processes and counts sequence data, useful for model input preparation.
   - `splitter.py`: Splits datasets into training, validation, and test sets, also limiting the sequence lengths.

3. **Models/**  
   This directory contains the architecture implementations of different models along with training,inference & hyper parameter tuning scripts .
   - `SineKAN/`: Contains the implementation of the **S-KANformer** model.
   - `vanilla/`: Contains the vanilla version of the model architecture used as a baseline for comparison.


