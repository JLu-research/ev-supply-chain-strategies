import pandas as pd
import numpy as np
from pyomo.environ import *
import re

# Import input files
material_share = pd.read_excel('data/material_share.xlsx')
mineral = pd.read_excel('data/mineral.xlsx')
component = pd.read_excel('data/component.xlsx')
specific_energy = pd.read_excel('data/specific_energy.xlsx')

# Define the range of years for the analysis
years = list(range(2025, 2036))

# Define battery chemistries and vehicle types
## Separate battery_type lists are used to group chemistries by the materials they use,
## making it easier to assign material-specific variables and constraints in the optimization model.

battery_types= ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811','SIB',
                'LFP_siliconaam', 'NCA_siliconaam', 'NMC111_siliconaam', 'NMC532_siliconaam',
                'NMC622_siliconaam', 'NMC811_siliconaam']

battery_types_LIB = ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811',
                     'LFP_siliconaam', 'NCA_siliconaam', 'NMC111_siliconaam', 'NMC532_siliconaam',
                     'NMC622_siliconaam', 'NMC811_siliconaam']

battery_types_LIB_graphiteaam = ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']

battery_types_LIB_siliconaam = ['LFP_siliconaam', 'NCA_siliconaam', 'NMC111_siliconaam',
                                'NMC532_siliconaam', 'NMC622_siliconaam', 'NMC811_siliconaam']

battery_types_SIB = ['SIB']

vehicle_types = ['BEV', 'PHEV']

# Store annual optimization results
results = []

# Optimization model
for year in years:
    model = ConcreteModel()

    # Decision variables: Amount of each eligible material/component for each chemistry and vehicle type
    model.bauxite_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.aluminum_hydroxide_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.aluminum_sulfate_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.cobalt_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.cobalt_sulfate_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.graphite_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.lithium_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.lithium_hydroxide_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.lithium_carbonate_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.manganese_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.manganese_sulfate_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.nickel_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.nickel_sulfate_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.cam_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.graphite_aam_eligible = Var(vehicle_types, battery_types_LIB_graphiteaam, within=NonNegativeReals)
    model.silicon_aam_eligible = Var(vehicle_types, battery_types_LIB_siliconaam, within=NonNegativeReals)
    model.sib_aam_eligible = Var(vehicle_types, battery_types_SIB, within=NonNegativeReals)
    model.electrolyte_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.separator_eligible = Var(vehicle_types, battery_types, within=NonNegativeReals)
    model.battery_eligible_production = Var(vehicle_types, battery_types, within=NonNegativeReals)



    # Precursor material requirement constraints
    for chem in battery_types:
        for veh in vehicle_types:

            model.add_component(f'{veh.lower()}_{chem.lower()}_aluminum_hydroxide_constraint',
                                Constraint(expr=
                                    model.aluminum_hydroxide_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'aluminum_hydroxide', f'{veh}_{chem}'].values[0]))

            model.add_component(f'{veh.lower()}_{chem.lower()}_aluminum_sulfate_constraint',
                                Constraint(expr=
                                    model.aluminum_sulfate_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'aluminum_sulfate', f'{veh}_{chem}'].values[0]))


            model.add_component(f'{veh.lower()}_{chem.lower()}_cobalt_sulfate_constraint',
                                Constraint(expr=
                                    model.cobalt_sulfate_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'cobalt_sulfate', f'{veh}_{chem}'].values[0]))

            model.add_component(f'{veh.lower()}_{chem.lower()}_lithium_hydroxide_constraint',
                                Constraint(expr=
                                    model.lithium_hydroxide_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'lithium_hydroxide', f'{veh}_{chem}'].values[0]))

            model.add_component(f'{veh.lower()}_{chem.lower()}_lithium_carbonate_constraint',
                                Constraint(expr=
                                    model.lithium_carbonate_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'lithium_carbonate', f'{veh}_{chem}'].values[0]))

            model.add_component(f'{veh.lower()}_{chem.lower()}_manganese_sulfate_constraint',
                                Constraint(expr=
                                    model.manganese_sulfate_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'manganese_sulfate', f'{veh}_{chem}'].values[0]))

            model.add_component(f'{veh.lower()}_{chem.lower()}_nickel_sulfate_constraint',
                                Constraint(expr=
                                    model.nickel_sulfate_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'nickel_sulfate', f'{veh}_{chem}'].values[0]))

    # Precursor material supply constraints
    model.add_component('aluminum_hydroxide_eligible_supply',
                        Constraint(expr=sum(model.aluminum_hydroxide_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'aluminum_hydroxide_eligible'].values[0]))

    model.add_component('aluminum_sulfate_eligible_supply',
                        Constraint(expr=sum(model.aluminum_sulfate_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'aluminum_sulfate_eligible'].values[0]))

    model.add_component('cobalt_sulfate_eligible_supply',
                        Constraint(expr=sum(model.cobalt_sulfate_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'cobalt_sulfate_eligible'].values[0]))

    model.add_component('lithium_hydroxide_eligible_supply',
                        Constraint(expr=sum(model.lithium_hydroxide_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'lithium_hydroxide_eligible'].values[0]))

    model.add_component('lithium_carbonate_eligible_supply',
                        Constraint(expr=sum(model.lithium_carbonate_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'lithium_carbonate_eligible'].values[0]))

    model.add_component('manganese_sulfate_eligible_supply',
                        Constraint(expr=sum(model.manganese_sulfate_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'manganese_sulfate_eligible'].values[0]))

    model.add_component('nickel_sulfate_eligible_supply',
                        Constraint(expr=sum(model.nickel_sulfate_eligible[veh, chem] for veh in vehicle_types for chem in battery_types) <= mineral.loc[mineral['year'] == year, 'nickel_sulfate_eligible'].values[0]))



    # Component requirement constraints
    ## cathode active materials and separators
    for chem in battery_types:
        for veh in vehicle_types:
            model.add_component(f'{veh.lower()}_{chem.lower()}_cam_constraint',
                                Constraint(expr=
                                    model.cam_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'cam', f'{veh}_{chem}'].values[0]))

            model.add_component(f'{veh.lower()}_{chem.lower()}_separator_constraint',
                                Constraint(expr=
                                    model.separator_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'separator', f'{veh}_{chem}'].values[0]))

    ## electrolytes
    for chem in battery_types_LIB:
        for veh in vehicle_types:

            model.add_component(f'{veh.lower()}_{chem.lower()}_electrolyte_constraint',
                                Constraint(expr=
                                    model.electrolyte_eligible[veh, chem] ==
                                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'electrolyte', f'{veh}_{chem}'].values[0]
                                          )
                               )

    ## graphite aam
    for chem in battery_types_LIB_graphiteaam:
        for veh in vehicle_types:

            model.add_component(f'{veh.lower()}_{chem.lower()}_graphite_aam_constraint',
                Constraint(expr=
                    model.graphite_aam_eligible[veh, chem] ==
                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'aam', f'{veh}_{chem}'].values[0]
                          )
                )


    ## silicon aam
    for chem in battery_types_LIB_siliconaam:
        for veh in vehicle_types:

            model.add_component(f'{veh.lower()}_{chem.lower()}_silicon_aam_constraint',
                Constraint(expr=
                    model.silicon_aam_eligible[veh, chem] ==
                    model.battery_eligible_production[veh, chem] * material_share.loc[material_share['material'] == 'aam', f'{veh}_{chem}'].values[0]
                          )
                )


    ## sodium-ion battery (SIB) anode active materials & SIB electrolytes
    for veh in vehicle_types:

        model.add_component(f'{veh.lower()}_sib_aam_constraint',
                            Constraint(expr=
                                model.sib_aam_eligible[veh, 'SIB']  ==
                                model.battery_eligible_production[veh, 'SIB'] * material_share.loc[material_share['material'] == 'aam', f'{veh}_SIB'].values[0]))

        model.add_component(f'{veh.lower()}_sib_electrolyte_constraint',
                            Constraint(expr=
                                model.electrolyte_eligible[veh, 'SIB'] ==
                                model.battery_eligible_production[veh, 'SIB'] * material_share.loc[material_share['material'] == 'electrolyte', f'{veh}_SIB'].values[0]))


    # Component supply constraints
    ## cathode active materials
    for chem in battery_types_LIB_graphiteaam:
        model.add_component(
            f'{chem.lower()}_cam_eligible_supply',
            Constraint(
                expr=sum(model.cam_eligible[veh, var] for veh in vehicle_types for var in [chem, f'{chem}_siliconaam']) <=
                     component.loc[component['year'] == year, f'{chem}_eligible'].values[0]
            )
        )

    model.sib_cam_eligible_supply = Constraint(expr=
        sum(model.cam_eligible[veh, 'SIB'] for veh in vehicle_types)
        <= component.loc[component['year'] == year, 'SIB_eligible'].values[0]
    )


    ## anode active materials
    model.graphite_aam_eligible_supply = Constraint(expr=
        sum(model.graphite_aam_eligible[veh, chem] for veh in vehicle_types for chem in battery_types_LIB_graphiteaam)
        <= component.loc[component['year'] == year, 'graphite_aam_eligible'].values[0]
    )

    model.silicon_aam_eligible_supply = Constraint(expr=
        sum(model.silicon_aam_eligible[veh, chem] for veh in vehicle_types for chem in battery_types_LIB_siliconaam)
        == component.loc[component['year'] == year, 'silicon_aam_eligible'].values[0]
    )

    model.sib_aam_eligible_supply = Constraint(expr=
        sum(model.sib_aam_eligible[veh, 'SIB'] for veh in vehicle_types)
        <= component.loc[component['year'] == year, 'SIB_aam_eligible'].values[0]
    )



    ## separators
    model.separator_eligible_supply = Constraint(expr=
        sum(model.separator_eligible[veh, chem] for veh in vehicle_types for chem in battery_types)
        <= component.loc[component['year'] == year, 'separator_eligible'].values[0]
    )


    ## electrolytes
    model.electrolyte_eligible_supply = Constraint(expr=
        sum(model.electrolyte_eligible[veh, chem] for veh in vehicle_types for chem in battery_types_LIB)
        <= component.loc[component['year'] == year, 'electrolyte_eligible'].values[0]
    )

    model.sib_electrolyte_eligible_supply = Constraint(expr=
        sum(model.electrolyte_eligible[veh, 'SIB'] for veh in vehicle_types)
        <= component.loc[component['year'] == year, 'SIB_electrolyte_eligible'].values[0]
    )


    # Objective: Maximize eligible battery production
    model.objective = Objective(expr=
        sum(model.battery_eligible_production[veh, chem] for veh in vehicle_types for chem in battery_types),
        sense=maximize
    )

    # Solve the model
    solver = SolverFactory('glpk')
    result = solver.solve(model, tee=False)

    # Collect results
    results.append({
        'year': year,
        **{f'{veh}_{chem}_battery_production': model.battery_eligible_production[veh, chem].value for veh in vehicle_types for chem in battery_types}
    })

# Combine annual results into a DataFrame
results_df = pd.DataFrame(results)


# Convert to GWh for each battery type/vehicle type
df = results_df.copy()
df = pd.merge(df, specific_energy, on='year')

# Calculate GWh for each vehicle type and battery chemistry
for veh in vehicle_types:
    for chem in battery_types:
        df[f'{veh}_{chem}_GWh'] = (
            df[f'{veh}_{chem}_battery_production'] * df[f'{veh}_{chem}_specificenergy'] / 1e9 * 1000
        )
# Aggregate GWh across vehicle types (BEV + PHEV) for each battery chemistry
for chem in battery_types:
    df[f'{chem}_GWh'] = df[f'BEV_{chem}_GWh'] + df[f'PHEV_{chem}_GWh']

# Calculate total GWh production per year (all chemistries)
total_GWh = df[[f'{chem}_GWh' for chem in battery_types]].sum(axis=1)

# Compute the production share of each chemistry per year
for chem in battery_types:
    df[f'{chem}_share'] = df[f'{chem}_GWh'] / total_GWh


# -------- Battery chemistry market share (LFP, NCA, NMC, SIB) --------

df_share = df[['year'] + [f'{chem}_share' for chem in battery_types]].copy()

# Combine graphiteaam and siliconaam
def combine_shares(df_share, bases):
    for base in bases:
        cols = [col for col in df_share.columns if re.fullmatch(f'{base}(_siliconaam)?_share', col)]
        if len(cols) > 1:
            df_share[f'{base}_share'] = df_share[cols].sum(axis=1)
    return df_share

bases = ['LFP', 'NCA', 'SIB', 'NMC111', 'NMC532', 'NMC622', 'NMC811']
df_share = combine_shares(df_share, bases)

# Sum all NMC* shares to create a single NMC_share column
df_share['NMC_share'] = df_share[['NMC111_share', 'NMC532_share', 'NMC622_share', 'NMC811_share']].sum(axis=1)

df_chem_share = df_share[['year', 'LFP_share', 'NCA_share', 'NMC_share', 'SIB_share']].copy()

# Convert shares to integer percentages (no decimal places)
df_chem_share.loc[:, df_chem_share.columns != 'year'] = (
    (df_chem_share.loc[:, df_chem_share.columns != 'year'] * 100).round(0).astype(int)
)


# -------- Anode material market share (Graphite AAM vs. Silicon AAM) --------
df_aam_share = df[['year'] + [f'{chem}_share' for chem in battery_types]].copy()

# Identify silicon-AAM columns (contain 'siliconaam'); graphite-AAM are all others
silicon_cols = [col for col in df_aam_share if 'siliconaam' in col]
graphite_cols = ['LFP_share', 'NCA_share', 'NMC111_share', 'NMC532_share', 'NMC622_share', 'NMC811_share', 'SIB_share']

# Sum market share for graphite-AAM and silicon-AAM chemistries
df_aam_share['graphite_aam'] = df_aam_share[graphite_cols].sum(axis=1)
df_aam_share['silicon_aam'] = df_aam_share[silicon_cols].sum(axis=1)

df_aam_share = df_aam_share[['year', 'graphite_aam', 'silicon_aam']]

# Convert shares to integer percentages (no decimal places)
df_aam_share.loc[:, df_aam_share.columns != 'year'] = (
    (df_aam_share.loc[:, df_aam_share.columns != 'year'] * 100).round(0).astype(int)
)

print(df_chem_share[df_chem_share['year']==2035])
print(df_aam_share[df_aam_share['year']==2035])
