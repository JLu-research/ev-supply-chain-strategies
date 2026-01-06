# ev-supply-chain-strategies

The code in this repository aims to estimate material supply-demand gaps in the US electric vehicle (EV) battery supply chain and to maximize EV battery production while accounting for constraints on material availability in the US. The modeling framework, assumptions, and data sources are described in detail in the Methods section and Supplementary Information of the associated paper. 

## 1. System requirements
The analysis was run on Python 3.9.7. The following Python libraries are required: `numpy`, `pandas`, `scipy`,  `pyomo`, `os`, and `re`. The script has been tested on MacOS. No special hardware is required. A standard laptop/desktop is sufficient.

## 2. Installation guide
All files can be directly downloaded from https://github.com/JLu-research/ev-supply-chain-strategies.git. Installation typically takes less than one minute.

## 3. Demo
Data files are included in the `data/` folder. These files allow users to generate material supply-demand gap estimates in the *“Supply Expansion Scenario”* and the market share results reported in the *“Switch to Alternative Battery Chemistries Scenario”* described in the paper. 

To generate material supply-demand gaps, users should open the script `Core_Analysis.py` in the Python environment. Running this script will print a summary table (`df_final`) that reports, for each material, demand; US existing & advanced-stage supply; US early-stage supply; imports from Canada and Mexico at historical average levels and from planned advanced-stage projects; imports from Canada and Mexico from planned early-stage projects; imports from other free-trade agreement countries; imports from foreign entity of concern countries; import from other countries; US recycling supply; and the resulting material supply-demand gaps in 2035. The script typically completes in less than one minute on a standard desktop or laptop.

To generate market share results, users should open the script `Battery_Optimization_Model.py` in the Python environment. Running this script will print two tables: one (`df_chem_share`) shows the battery chemistry market share in 2035, and the other (`df_aam_share`) shows the anode active material market share in 2035. The script typically completes in less than one minute on a standard desktop or laptop.

## 4. Instructions for use
Users interested in applying the scripts to other regions or countries may modify the input data, parameters, and scripts to reflect local conditions. For example, users can substitute data and assumptions such as EV demand trajectories, battery chemistry market shares, battery energy density, domestic production levels, and import sources and volumes. Results generated under such adaptations should be interpreted based on relevant regional and country-specific contexts. 
