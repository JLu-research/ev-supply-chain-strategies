# ev-supply-chain-strategies

The code in this repository aims to maximize electric vehicle battery production while accounting for constraints on material availability in the US. Details of the model and data can be found in the Methods section and Supplementary Information of the paper. The script and data files in this repository are intended for reviewer use only.

## 1. System requirements
The analysis was run on Python 3.9.7. The following python libraries are required: `numpy`, `pandas`, `pyomo`, and `re`. The script has been tested on MacOS. No special hardware is required. A standard laptop/desktop is sufficient.

## 2. Installation guide
All files can be directly downloaded from https://github.com/JLu-research/ev-supply-chain-strategies.git. Installation typically takes less than one minute.

## 3. Demo
Sample data files are included in the `data/` folder. These files allow users to generate the market share results reported in the *“Switch to Alternative Battery Chemistry”* scenario described in the paper. 

To run the demo, users should open the script `Battery_Optimization_Model.py` in the Python environment. If necessary, users can modify the script to specify the correct path to the sample data files. 

Running the script will print two tables: one (`df_chem_share`) shows the battery chemistry market share in 2035, and the other (`df_aam_share`) shows the anode material market share in 2035. For the demo dataset, the script typically completes in less than one minute on a standard desktop or laptop.

## 4. Instructions for use
To use the script with custom data, users should prepare input files that follow the same structure and column names as the sample files in the `data/` folder. The sample files can be replaced with user-generated files, or the script can be modified to reference different file paths. When the script is executed, the output DataFrames will contain market share results for the user-provided dataset.
