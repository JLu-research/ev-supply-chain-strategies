import pandas as pd
import numpy as np
import os
from scipy.optimize import brentq

PARAM_FILE = 'data/core_analysis/input.xlsx'

def read_sheet(sheet_name, **kwargs):
    return pd.read_excel(PARAM_FILE, sheet_name=sheet_name, **kwargs)

# Input files used in the analysis
SHEETS = {
    'EV_sales':             'EV_sales',
    'battery_wt_year':      'battery_wt_year',
    'specific_energy_year': 'specific_energy_year',
    'market_share':         'market_share',
    'aam_market_share':     'aam_market_share',
    'recycling_rate':       'recycling_rate',
    'recycling_portion':    'recycling_portion',
    'recycling_limit':      'recycling_limit',
    'downstream_material':  'downstream_material',
    'midstream_material':   'midstream_material',
    'upstream_material':    'upstream_material',
    'pyro_recycling':       'pyro_recycling',
    'hydro_recycling':      'hydro_recycling',
    'direct_recycling':     'direct_recycling',
}

inputs = {name: read_sheet(sheet) for name, sheet in SHEETS.items()}

EV_sales             = inputs['EV_sales']
battery_wt_year      = inputs['battery_wt_year']
specific_energy_year = inputs['specific_energy_year']
market_share         = inputs['market_share']
aam_market_share     = inputs['aam_market_share']
recycling_rate       = inputs['recycling_rate']
recycling_portion    = inputs['recycling_portion']
recycling_limit      = inputs['recycling_limit']
downstream_material  = inputs['downstream_material']
midstream_material   = inputs['midstream_material']
upstream_material    = inputs['upstream_material']
pyro_recycling       = inputs['pyro_recycling']
hydro_recycling      = inputs['hydro_recycling']
direct_recycling     = inputs['direct_recycling']


# ----- Estimate material demand ----- #

# Battery cell
demand = pd.merge(EV_sales, market_share, on='year')

for column in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811','SIB']:
    demand['BEV_' + column] = demand['BEV'] * demand[column]
    demand['PHEV_' + column] = demand['PHEV'] * demand[column]


for vehicle in ['BEV', 'PHEV']:
    for chemistry in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811','SIB']:

        demand_col = f'{vehicle}_{chemistry}'
        wt_col = f'{vehicle}_{chemistry}_wt'
        new_col = f'{demand_col}_ton'

        if demand_col in demand.columns:

            demand[new_col] = demand[demand_col] * battery_wt_year[wt_col]

cell_demand = demand[[col for col in demand.columns if col.endswith('_ton') or col == 'year']]


# Battery component demand
all_tables = []

for material in downstream_material['material'].unique():

    current_share = downstream_material[downstream_material['material'] == material].iloc[0]

    temp_df = demand[['year']].copy()
    temp_df['material'] = material

    for col in demand.columns:
        if col.endswith('_ton'):

            share_key = col.replace('_ton', '')

            if share_key in current_share:
                share_value = current_share[share_key]
                temp_df[col] = demand[col] * share_value

    all_tables.append(temp_df)

material_demand = pd.concat(all_tables, ignore_index=True)


## Electrolyte
### LIB electrolyte
lib_electrolyte_demand = (
    material_demand
    .loc[material_demand['material'].isin(['LIB electrolyte'])]
    .groupby('year', as_index=False)
    .sum()
)

### SIB electrolyte
sib_electrolyte_demand = (
    material_demand
    .loc[material_demand['material'].isin(['SIB electrolyte'])]
    .groupby('year', as_index=False)
    .sum()
)

## Separator
separator_demand = (
    material_demand
    .loc[material_demand['material'].isin(['Separator'])]
    .groupby('year', as_index=False)
    .sum()
)

## Cathode active material (CAM)
cam_demand = (
    material_demand
    .loc[material_demand['material'].isin(['Active Material'])]
    .groupby('year', as_index=False)
    .sum()
)

## LIB graphite anode active material (AAM)
graphite_aam_demand = (
    material_demand
    .loc[material_demand['material'].isin(['LIB AAM'])]
    .groupby('year', as_index=False)
    .sum()
)

graphite_aam_demand = graphite_aam_demand.merge(aam_market_share, on='year')

ton_cols = [col for col in graphite_aam_demand.columns if col.endswith('_ton')]

graphite_aam_demand[ton_cols] = (
    graphite_aam_demand[ton_cols]
    .mul(graphite_aam_demand['graphite_share'], axis=0)
)

graphite_aam_demand = graphite_aam_demand.drop(columns=['graphite_share'])


## LIB graphite-silicon AAM
silicon_aam_demand = (
    material_demand
    .loc[material_demand['material'].isin(['LIB AAM'])]
    .groupby('year', as_index=False)
    .sum()
)

silicon_aam_demand = silicon_aam_demand.merge(aam_market_share, on='year')

ton_cols = [col for col in silicon_aam_demand.columns if col.endswith('_ton')]

silicon_aam_demand[ton_cols] = (
    silicon_aam_demand[ton_cols]
    .mul(1 - silicon_aam_demand['graphite_share'], axis=0)
)

silicon_aam_demand = silicon_aam_demand.drop(columns=['graphite_share'])


## SIB AAM
sib_aam_demand = (
    material_demand
    .loc[material_demand['material'] == 'SIB AAM']
    .groupby('year', as_index=False)
    .sum()
)


## Binder
binder_demand = (
    material_demand
    .loc[material_demand['material'] == 'Binder']
    .groupby('year', as_index=False)
    .sum()
)


# Refined material demand
## Lithium carbonate
lithium_carbonate_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Lithium carbonate (Li2CO3)']
    .iloc[0]
)

lithium_carbonate_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in lithium_carbonate_row.index and pd.notna(lithium_carbonate_row[chem]):
        share = lithium_carbonate_row[chem]
        lithium_carbonate_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        lithium_carbonate_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share

## Lithium hydroxide
lithium_hydroxide_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Lithium hydroxide (LiOH)']
    .iloc[0]
)

lithium_hydroxide_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in lithium_hydroxide_row.index and pd.notna(lithium_hydroxide_row[chem]):
        share = lithium_hydroxide_row[chem]
        lithium_hydroxide_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        lithium_hydroxide_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share

## Nickel sulfate
nickel_sulfate_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Nickel sulfate (NiSO4)']
    .iloc[0]
)

nickel_sulfate_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in nickel_sulfate_row.index and pd.notna(nickel_sulfate_row[chem]):
        share = nickel_sulfate_row[chem]
        nickel_sulfate_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        nickel_sulfate_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share


## Cobalt Sulfate
cobalt_sulfate_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Cobalt sulfate (CoSO4)']
    .iloc[0]
)

cobalt_sulfate_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in cobalt_sulfate_row.index and pd.notna(cobalt_sulfate_row[chem]):
        share = cobalt_sulfate_row[chem]
        cobalt_sulfate_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        cobalt_sulfate_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share

## Manganese sulfate
manganese_sulfate_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Manganese sulfate (MnSO4)']
    .iloc[0]
)

manganese_sulfate_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in manganese_sulfate_row.index and pd.notna(manganese_sulfate_row[chem]):
        share = manganese_sulfate_row[chem]
        manganese_sulfate_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        manganese_sulfate_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share

## Aluminum sulfate
aluminum_sulfate_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Aluminum sulfate (Al2(SO4)3)']
    .iloc[0]
)

aluminum_sulfate_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in aluminum_sulfate_row.index and pd.notna(aluminum_sulfate_row[chem]):
        share = aluminum_sulfate_row[chem]
        aluminum_sulfate_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        aluminum_sulfate_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share

## Aluminum hydroxide
aluminum_hydroxide_row = (
    midstream_material
    .loc[midstream_material['material'] == 'Aluminum hydroxide Al(OH)3']
    .iloc[0]
)

aluminum_hydroxide_demand = cam_demand[['year']].copy()

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
    if chem in aluminum_hydroxide_row.index and pd.notna(aluminum_hydroxide_row[chem]):
        share = aluminum_hydroxide_row[chem]
        aluminum_hydroxide_demand[f'BEV_{chem}_ton']  = cam_demand[f'BEV_{chem}_ton']  * share
        aluminum_hydroxide_demand[f'PHEV_{chem}_ton'] = cam_demand[f'PHEV_{chem}_ton'] * share

# Raw material demand
## Raw graphite
raw_graphite_demand1 = graphite_aam_demand.copy()

raw_graphite_demand1.loc[:, raw_graphite_demand1.columns != 'year'] *= (
    # Assuming 40% of graphite AAM is sourced from natural graphite and 60% from synthetic graphite
    0.4 * upstream_material.loc[
        upstream_material['mineral'] == 'Natural graphite', 'Graphite AAM'
    ].iloc[0]
    +
    0.6 * upstream_material.loc[
        upstream_material['mineral'] == 'Synthetic graphite', 'Graphite AAM'
    ].iloc[0]
)

raw_graphite_demand2 = silicon_aam_demand.copy()

raw_graphite_demand2.loc[:, raw_graphite_demand2.columns != 'year'] *= (
    # Assuming 90% graphite, 10% silicon in graphite-silicon aam
    0.9 * 0.4 * upstream_material.loc[
        upstream_material['mineral'] == 'Natural graphite', 'Graphite AAM'
    ].iloc[0]
    +
    0.9 * 0.6 * upstream_material.loc[
        upstream_material['mineral'] == 'Synthetic graphite', 'Graphite AAM'
    ].iloc[0]
)

raw_graphite_demand = (
    pd.concat([raw_graphite_demand1, raw_graphite_demand2], ignore_index=True)
      .groupby('year', as_index=False)
      .sum(numeric_only=True)
)

## Raw lithium
lithiumore_demand = (
    pd.concat([

        lithium_hydroxide_demand.assign(
            **{
                c: lithium_hydroxide_demand[c]
                   * upstream_material.loc[
                       upstream_material['mineral'] == 'Lithium ore',
                       'Lithium hydroxide'
                     ].iloc[0]
                for c in lithium_hydroxide_demand.columns
                if c != 'year'
            }
        ),

        lithium_carbonate_demand.assign(
            **{
                c: lithium_carbonate_demand[c]
                   * 0.5 # Assuming 50% coming from ore
                   * upstream_material.loc[
                       upstream_material['mineral'] == 'Lithium ore',
                       'Lithium carbonate'
                     ].iloc[0]
                for c in lithium_carbonate_demand.columns
                if c != 'year'
            }
        )
    ])
    .groupby('year', as_index=False)
    .sum()
)

lithiumbrine_demand = lithium_carbonate_demand.assign(
    **{
        c: lithium_carbonate_demand[c]
           * 0.5 # Assuming 50% coming from brine
           * upstream_material.loc[
               upstream_material['mineral'] == 'Lithium brine',
               'Lithium carbonate'
             ].iloc[0]
        for c in lithium_carbonate_demand.columns
        if c != 'year'
    }
)

raw_lithium_demand = (
    pd.concat([lithiumore_demand, lithiumbrine_demand])
    .groupby('year', as_index=False)
    .sum()
)

## Raw nickel
raw_nickel_demand = nickel_sulfate_demand.copy()

raw_nickel_demand.loc[
    :, raw_nickel_demand.columns != 'year'
] *= upstream_material.loc[
        upstream_material['mineral'] == 'Nickel',
        'Nickel sulfate'
    ].iloc[0]

## Raw cobalt
raw_cobalt_demand = cobalt_sulfate_demand.copy()

raw_cobalt_demand.loc[
    :, raw_cobalt_demand.columns != 'year'
] *= upstream_material.loc[
        upstream_material['mineral'] == 'Cobalt',
        'Cobalt sulfate'
    ].iloc[0]

# Raw manganese
raw_manganese_demand = manganese_sulfate_demand.copy()

raw_manganese_demand.loc[
    :, raw_manganese_demand.columns != 'year'
] *= upstream_material.loc[
        upstream_material['mineral'] == 'Manganese',
        'Manganese sulfate'
    ].iloc[0]

# Bauxite
bauxite_demand = aluminum_hydroxide_demand.copy()

bauxite_demand.loc[
    :, bauxite_demand.columns != 'year'] *= upstream_material.loc[
        upstream_material['mineral'] == 'Bauxite',
        'Aluminum hydroxide'
    ].iloc[0]

# ----- Estimate recycled material availability ----- #

# Prepare recycling inputs and capacity constraints
recycling_portions = recycling_portion['recycling_portion'].tolist()

use_capacity_limit = (recycling_limit.iloc[0, 0] == 1)

recycling_capacity_US = pd.read_excel('data/core_analysis/planned/all/recycling_US.xlsx')

recycling_rate = recycling_rate.merge(recycling_capacity_US, on='year', how='outer')
recycling_rate['recycling_capacity_rp_sum'] = recycling_rate['recycling_capacity_existing'].fillna(0)+recycling_rate['recycling_capacity_rp_US'].fillna(0)

# Construct battery cell demand by chemistry
battery_cell_demand = demand[[col for col in demand.columns if col.endswith('ton') or col == 'year']]

for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
    battery_cell_demand[chem] = battery_cell_demand[f'BEV_{chem}_ton'] + battery_cell_demand[f'PHEV_{chem}_ton']

eol_battery_input = battery_cell_demand[[col for col in battery_cell_demand.columns if col in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB'] or col == 'year']]

# Battery lifetime modeling using Weibull distributions
def fit_weibull_from_min_mode_max(a: float, d: float, c: float, q: float = 0.997):
    """Estimate Weibull shape and scale parameters from minimum, mode, and maximum lifetime assumptions."""
    if not (c > d > a):
        raise ValueError('Require c (max) > d (mode) > a (min).')
    Lq = -np.log(1.0 - q)

    def beta_from_alpha(alpha):
        return (c - a) / (Lq ** (1.0 / alpha))

    def f(alpha):
        if alpha <= 1.0:
            return np.inf
        beta = beta_from_alpha(alpha)
        mode = a + beta * ((alpha - 1.0) / alpha) ** (1.0 / alpha)
        return mode - d

    alpha = brentq(f, 1.001, 50.0)
    beta = beta_from_alpha(alpha)
    return alpha, beta

def weibull_yearly_kernel_from_params(a_min: int, alpha: float, beta: float, max_age: int = 30) -> pd.Series:
    """Convert a Weibull lifetime distribution into a discrete annual retirement kernel."""
    ages = np.arange(a_min, max_age + 1)
    cdf = np.zeros_like(ages, dtype=float)
    mask = ages >= a_min
    cdf[mask] = 1.0 - np.exp(-((ages[mask] - a_min) / beta) ** alpha)
    pdf = np.diff(np.insert(cdf, 0, 0.0))
    pdf = np.clip(pdf, 0, None)
    pdf /= pdf.sum()

    return pd.Series(pdf, index=ages,
                     name=f'weibull_shifted_a={a_min}_alpha={alpha:.3f}_beta={beta:.3f}')

def retired_evs_from_min_mode_max(
    sales_by_chem: pd.DataFrame,
    chem_specs: dict,
    q: float = 0.997,
    max_age: int = 30,
    year_col: str = 'year',
    skip_unknown: bool = True,
    aggregate_nmc: bool = False
) -> pd.DataFrame:
    """Apply chemistry-specific lifetime kernels to estimate annual end-of-life battery flows."""
    chem_cols = [c for c in sales_by_chem.columns if c != year_col]

    resolved_params = {}
    valid_cols = []
    unknown_cols = []

    for chem in chem_cols:
        cname = chem.upper()

        if chem in chem_specs:
            key = chem
        elif 'NMC' in cname and 'NMC*' in chem_specs:
            key = 'NMC*'
        else:
            unknown_cols.append(chem)
            continue

        spec = chem_specs[key]
        a, d, c = spec['min'], spec['mode'], spec['max']
        if not (c > d > a):
            raise ValueError(f'Invalid min/mode/max for {chem}: require max > mode > min.')

        resolved_params[chem] = (a, d, c)
        valid_cols.append(chem)

    if unknown_cols:
        print(f"No lifetime parameters for: {', '.join(unknown_cols)}. "
              f"Retirements for these columns will be set to 0 in the output.")

    if not valid_cols:
        y_min = int(sales_by_chem[year_col].min())
        y_max = int(sales_by_chem[year_col].max()) + max_age
        out_years = np.arange(y_min, y_max + 1, dtype=int)
        reti = pd.DataFrame(0.0, index=out_years, columns=chem_cols)
        return reti.reset_index().rename(columns={'index': 'year'})

    triplet_to_kernel = {}
    for triplet in sorted(set(resolved_params.values())):
        a, d, c = triplet
        alpha, beta = fit_weibull_from_min_mode_max(a, d, c, q=q)
        kernel = weibull_yearly_kernel_from_params(a_min=int(np.floor(a)), alpha=alpha, beta=beta, max_age=max_age)
        triplet_to_kernel[triplet] = kernel

    y_min = int(sales_by_chem[year_col].min())
    y_max = int(sales_by_chem[year_col].max()) + max_age
    out_years = np.arange(y_min, y_max + 1, dtype=int)

    reti = pd.DataFrame(0.0, index=out_years, columns=chem_cols)

    for _, row in sales_by_chem.iterrows():
        t = int(row[year_col])
        for chem in valid_cols:
            sold = float(row[chem])
            if sold == 0:
                continue
            kernel = triplet_to_kernel[resolved_params[chem]]
            for age, frac in kernel.items():
                reti.at[t + age, chem] += sold * frac

    reti = reti.reset_index().rename(columns={'index': 'year'})

    if aggregate_nmc:
        nmc_cols = [c for c in chem_cols if 'NMC' in c.upper()]
        nmc_cols = [c for c in nmc_cols if c in reti.columns]
        if nmc_cols:
            reti['NMC'] = reti[nmc_cols].sum(axis=1)
            reti.drop(columns=nmc_cols, inplace=True)

    return reti

## Battery lifetime
chem_specs = {
'LFP':  {'min': 1, 'mode': 9, 'max': 25},
'SIB':  {'min': 1, 'mode': 9, 'max': 25},
'NMC*': {'min': 1, 'mode': 8, 'max': 20},
'NCA':  {'min': 1, 'mode': 8, 'max': 20}
}

params = {}
for chem, spec in chem_specs.items():
    alpha, beta = fit_weibull_from_min_mode_max(spec['min'], spec['mode'], spec['max'], q=0.997)
    params[chem] = {'a': spec['min'], 'alpha': alpha, 'beta': beta}

eol_battery = retired_evs_from_min_mode_max(eol_battery_input, chem_specs, q=0.997, max_age = 100)

# Apply recycling rates with carryover and capacity constraints
recycling_rate_rp = recycling_rate[['year','recycling_rate','recycling_capacity_rp_sum']]
eol_battery_rp = eol_battery.merge(recycling_rate_rp, on='year')

carryover = {battery: 0 for battery in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']}

recycled_rows = []
non_recycled_rows = []
available_eol_rows = []

for index, row in eol_battery_rp.iterrows():
    recycled_data = {'year': row['year']}
    non_recycled_data = {'year': row['year']}
    available_eol_data = {'year': row['year']}

    total_recycled = 0.0

    for battery in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
        current_eol = row[battery] + carryover[battery]
        recycled = current_eol * row['recycling_rate']
        recycled_data[battery] = recycled
        total_recycled += recycled
        available_eol_data[battery] = current_eol

    if use_capacity_limit and total_recycled > row['recycling_capacity_rp_sum']:
        adjustment_ratio = row['recycling_capacity_rp_sum'] / total_recycled
        for battery in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
            recycled_data[battery] *= adjustment_ratio

    for battery in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
        current_eol = available_eol_data[battery]
        non_recycled = current_eol - recycled_data[battery]

        carryover[battery] = non_recycled
        non_recycled_data[battery] = non_recycled

    recycled_rows.append(recycled_data)
    non_recycled_rows.append(non_recycled_data)
    available_eol_rows.append(available_eol_data)


recycled_df = pd.DataFrame(recycled_rows)
non_recycled_df = pd.DataFrame(non_recycled_rows)
available_eol_df = pd.DataFrame(available_eol_rows)


# Allocate recycled batteries across recycling technologies and quantify recycled material availability
data_list = []

for index, row in recycled_df.iterrows():
    for battery_type in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
        battery_values = [
            row[battery_type] * recycling_portions[0],
            row[battery_type] * recycling_portions[1],
            row[battery_type] * recycling_portions[2]
        ]

        pyro_multiplier = pyro_recycling.set_index('output')[battery_type]
        hydro_multiplier = hydro_recycling.set_index('output')[battery_type]
        direct_multiplier = direct_recycling.set_index('output')[battery_type]

        for recycling_type, multiplier, battery_value in zip(
            ['pyro_recycling', 'hydro_recycling', 'direct_recycling'],
            [pyro_multiplier, hydro_multiplier, direct_multiplier],
            battery_values):

            for output in multiplier.index:
                data_list.append({
                    'year': row['year'],
                    'battery_type': battery_type,
                    'recycling_type': recycling_type,
                    'output': output,
                    'quantity_ton': battery_value * multiplier[output]
                })

all_recycling = pd.DataFrame(data_list)

for battery_type in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
    all_recycling.loc[(all_recycling['battery_type'] == battery_type) & (all_recycling['output']== 'CAM'), 'output'] = f'{battery_type} CAM'

all_recycling_grouped = (
    all_recycling
        .groupby(['year', 'output'], as_index=False)['quantity_ton']
        .sum()
)

recycling_output = (
    all_recycling_grouped
        .pivot(index='year', columns='output', values='quantity_ton')
        .rename_axis(None, axis=1)
        .reset_index()
)

# Quantify avoided midstream and upstream material production
for cam_col in [c for c in recycling_output.columns if c.endswith(' CAM')]:
    chem = cam_col.replace(' CAM', '')
    if chem not in midstream_material.columns:
        continue
    for _, row in midstream_material.iterrows():
        mat = row['material']
        factor = row[chem]
        if pd.isna(factor) or factor == 0:
            continue
        new_col = f'{cam_col} {mat}'
        recycling_output[new_col] = recycling_output[cam_col] * factor

cols_to_drop = []

for mat in [
    'Nickel sulfate',
    'Cobalt sulfate',
    'Manganese sulfate',
    'Lithium hydroxide',
    'Lithium carbonate',
    'Aluminum hydroxide',
    'Aluminum sulfate'
]:
    mat_cols = [c for c in recycling_output.columns if mat in c]
    recycling_output[f'{mat} sum'] = recycling_output[mat_cols].sum(axis=1)

    cols_to_drop.extend(mat_cols)

recycling_output = recycling_output.drop(columns=cols_to_drop)
recycling_output = recycling_output.rename(columns=lambda c: c.replace(' sum', ''))


ms = upstream_material.set_index('mineral')

for mat_col in [c for c in recycling_output.columns if c in ms.columns]:

    if mat_col == 'Lithium carbonate':
        li_carbonate = recycling_output['Lithium carbonate']

        carbonate_ore_factor = ms.loc['Lithium ore', 'Lithium carbonate']
        carbonate_brine_factor = ms.loc['Lithium brine', 'Lithium carbonate']

        # Assuming 50% of lithium carbonte is sourced from ores and the rest is from brines.
        recycling_output['Lithium_ore_carbonate'] = li_carbonate * 0.5 * carbonate_ore_factor
        recycling_output['Lithium_brine_carbonate'] = li_carbonate * 0.5 * carbonate_brine_factor
        continue

    if mat_col == 'Lithium hydroxide':
        li_hydroxide = recycling_output['Lithium hydroxide']

        hydroxide_ore_factor = ms.loc['Lithium ore', 'Lithium hydroxide']

        recycling_output['Lithium_ore_hydroxide'] = li_hydroxide * hydroxide_ore_factor
        continue

    if mat_col == 'Graphite AAM':
        graphite_aam = recycling_output['Graphite AAM']

        nat_factor = ms.loc['Natural graphite', 'Graphite AAM']
        synth_factor = ms.loc['Synthetic graphite', 'Graphite AAM']

        # Assuming 40% of graphite AAM is sourced from natural graphite and 60% from synthetic graphite
        recycling_output['Natural graphite']   = graphite_aam * 0.4 * nat_factor
        recycling_output['Synthetic graphite'] = graphite_aam * 0.6 * synth_factor
        continue

    for mineral, factor in ms[mat_col].items():
        if pd.isna(factor) or factor == 0:
            continue

        recycling_output[mineral] = recycling_output.get(mineral, 0) + recycling_output[mat_col] * factor

recycling_output['Raw lithium'] = recycling_output['Lithium_ore_carbonate'] + recycling_output['Lithium_brine_carbonate'] + recycling_output['Lithium_ore_hydroxide']
recycling_output['Raw graphite'] = recycling_output['Natural graphite'] + recycling_output['Synthetic graphite']

recycling_output = recycling_output.drop(columns=['Lithium ore',
                              'Lithium brine',
                              'Natural graphite',
                              'Synthetic graphite'], errors='ignore')

recycling_output = recycling_output.rename(columns={
    'Nickel': 'Raw nickel',
    'Manganese': 'Raw manganese',
    'Cobalt': 'Raw cobalt'
})


# ----- Quantify material supply-demand gaps ----- #

# User-defined year
year = 2035

##### Demand #####
result = pd.read_excel('data/core_analysis/template.xlsx')
result.rename(columns={'US': 'US_mature',
                       'CaMex': 'CaMex_mature'}, inplace=True)
result['US_unmature'] = pd.NA
result['CaMex_unmature'] = pd.NA

demand_name = ['bauxite', 'raw_cobalt', 'raw_graphite', 'raw_lithium', 'raw_manganese', 'raw_nickel',
               'aluminum_hydroxide', 'aluminum_sulfate', 'cobalt_sulfate',
               'lithium_carbonate', 'lithium_hydroxide', 'manganese_sulfate', 'nickel_sulfate',
               'cam', 'graphite_aam', 'silicon_aam', 'sib_aam',
               'lib_electrolyte', 'sib_electrolyte','separator', 'binder',
               'cell']

demand_data = {}

for name in demand_name:
    var = f'{name}_demand'
    if var in locals():
        demand_data[name] = locals()[var]

demand_results = {}
for name in demand_name:
    df = demand_data.get(name)

    if df is not None:
        df_year = df[df['year'] == year]
        total_demand = df_year.drop(columns=['year']).sum().sum()
        demand_results[name] = total_demand

# Write demand into the reporting template
result.loc[result['Material'] == 'Bauxite', 'Demand'] = demand_results.get('bauxite')
result.loc[result['Material'] == 'Raw cobalt', 'Demand'] = demand_results.get('raw_cobalt')
result.loc[result['Material'] == 'Raw graphite', 'Demand'] = demand_results.get('raw_graphite')
result.loc[result['Material'] == 'Raw lithium', 'Demand'] = demand_results.get('raw_lithium')
result.loc[result['Material'] == 'Raw manganese', 'Demand'] = demand_results.get('raw_manganese')
result.loc[result['Material'] == 'Raw nickel', 'Demand'] = demand_results.get('raw_nickel')
result.loc[result['Material'] == 'Aluminum hydroxide', 'Demand'] = demand_results.get('aluminum_hydroxide')
result.loc[result['Material'] == 'Aluminum sulfate', 'Demand'] = demand_results.get('aluminum_sulfate')
result.loc[result['Material'] == 'Cobalt sulfate', 'Demand'] = demand_results.get('cobalt_sulfate')
result.loc[result['Material'] == 'Lithium carbonate', 'Demand'] = demand_results.get('lithium_carbonate')
result.loc[result['Material'] == 'Lithium hydroxide', 'Demand'] = demand_results.get('lithium_hydroxide')
result.loc[result['Material'] == 'Manganese sulfate', 'Demand'] = demand_results.get('manganese_sulfate')
result.loc[result['Material'] == 'Nickel sulfate', 'Demand'] = demand_results.get('nickel_sulfate')
result.loc[result['Material'] == 'Graphite AAM', 'Demand'] = demand_results.get('graphite_aam')
result.loc[result['Material'] == 'Silicon AAM', 'Demand'] = demand_results.get('silicon_aam')
result.loc[result['Material'] == 'LIB electrolyte', 'Demand'] = demand_results.get('lib_electrolyte')
result.loc[result['Material'] == 'Separator', 'Demand'] = demand_results.get('separator')
result.loc[result['Material'] == 'Binder', 'Demand'] = demand_results.get('binder')
result.loc[result['Material'] == 'SIB AAM', 'Demand'] = demand_results.get('sib_aam')
result.loc[result['Material'] == 'SIB electrolyte', 'Demand'] = demand_results.get('sib_electrolyte')

#  Write CAM demand into the reporting template
cam_demand = demand_data.get('cam')
cam_demand_year = cam_demand[cam_demand['year'] == year]

for battery in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
    columns_to_sum = [col for col in cam_demand_year.columns if battery in col]
    total_demand = cam_demand_year[columns_to_sum].sum().sum()
    material_name = f'{battery} CAM'
    result.loc[result['Material'] == material_name, 'Demand'] = total_demand

# Write cell demand into the reporting template
cell_demand = demand_data.get('cell')
cell_demand_year = cell_demand[cell_demand['year'] == year]

for battery in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
    columns_to_sum = [col for col in cell_demand_year.columns if battery in col]
    total_demand = cell_demand_year[columns_to_sum].sum().sum()
    material_name = f'{battery} cell'
    result.loc[result['Material'] == material_name, 'Demand'] = total_demand


##### Recycling #####
recycling_output_year = recycling_output[recycling_output['year'] == year]

for column in recycling_output.columns:
    if column != 'year':
        value = recycling_output_year[column].values[0] if not recycling_output_year.empty else None
        result.loc[result['Material'] == column, 'US recycling'] = value

##### Existing US supply and imports #####
supply_bl = pd.read_excel('data/core_analysis/template.xlsx')

supply_name = ['bauxite', 'raw_cobalt', 'raw_lithium', 'raw_manganese', 'raw_graphite', 'raw_nickel',
               'aluminum_hydroxide', 'aluminum_sulfate', 'cobalt_sulfate',
               'lithium_carbonate', 'lithium_hydroxide', 'manganese_sulfate', 'nickel_sulfate',
               'lfp_cam', 'nickel_cam',
               'graphite_aam', 'silicon_aam',
               'lib_electrolyte', 'separator', 'binder',
               'cell']

supply_data = {}
for name in supply_name:
    supply_data[name] = {}
    for country in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
        file_path = f'data/core_analysis/supply/{name}_{country}.xlsx'
        if os.path.exists(file_path):
            try:
                supply_data[name][country] = pd.read_excel(file_path)
            except Exception:
                pass

supply_results = {}
for name in supply_name:
    supply_results[name] = {}
    for country in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
        df = supply_data.get(name, {}).get(country, None)
        if df is not None:
            df_year = df[df['year'] == year]
            value = df_year['quantity_ton_battery'].values[0]
            supply_results[name][country] = value

# Write  supply into the reporting template
for country in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
    supply_bl.loc[supply_bl['Material'] == 'Bauxite', country] = supply_results.get('bauxite', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Raw cobalt', country] = supply_results.get('raw_cobalt', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Raw lithium', country] = supply_results.get('raw_lithium', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Raw manganese', country] = supply_results.get('raw_manganese', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Raw graphite', country] = supply_results.get('raw_graphite', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Raw nickel', country] = supply_results.get('raw_nickel', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Aluminum hydroxide', country] = supply_results.get('aluminum_hydroxide', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Aluminum sulfate', country] = supply_results.get('aluminum_sulfate', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Cobalt sulfate', country] = supply_results.get('cobalt_sulfate', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Lithium carbonate', country] = supply_results.get('lithium_carbonate', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Lithium hydroxide', country] = supply_results.get('lithium_hydroxide', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Manganese sulfate', country] = supply_results.get('manganese_sulfate', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Nickel sulfate', country] = supply_results.get('nickel_sulfate', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'LFP CAM', country] = supply_results.get('lfp_cam', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Graphite AAM', country] = supply_results.get('graphite_aam', {}).get(country, None)
    supply_bl.loc[supply_bl['Material'] == 'Silicon AAM', country] = supply_results.get('silicon_aam', {}).get(country, None)

# Write nickel CAMs into the reporting template
for cls in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
    try:
        nickel_cam_df = supply_data.get('nickel_cam')[cls]
    except KeyError:
        continue

    nickel_year_row = nickel_cam_df.loc[nickel_cam_df['year'] == year]
    for chem in ['NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
        material_name = f'{chem} CAM'
        value = nickel_year_row[chem].values[0]
        supply_bl.loc[supply_bl['Material'] == material_name, cls] = value

# Write electrolyte into the reporting templte
for country in ['CaMex', 'FTA', 'FEOC', 'nonFTA']:
    supply_bl.loc[supply_bl['Material'] == 'LIB electrolyte', country] = supply_results.get('lib_electrolyte', {}).get(country, None)

electrolyte_US = supply_data.get('lib_electrolyte')['US']
supply_bl.loc[supply_bl['Material'] == 'LIB electrolyte', 'US'] = (
    electrolyte_US.loc[
        electrolyte_US['year'] == year,
        [
            'LFP electrolyte', 'NCA electrolyte',
            'NMC111 electrolyte', 'NMC532 electrolyte', 'NMC622 electrolyte', 'NMC811 electrolyte'
        ]
    ]
    .sum(axis=1, skipna=True)
    .values[0]
)

supply_bl.loc[supply_bl['Material'] == 'SIB electrolyte', 'US'] = electrolyte_US.loc[electrolyte_US['year'] == year, 'SIB electrolyte'].values[0]

# Write separator into the reporting template
for country in ['CaMex', 'FTA', 'FEOC', 'nonFTA']:
    df = supply_data.get('separator')[country]
    supply_bl.loc[supply_bl['Material'] == 'Separator', country] = df.loc[df['year'] == year, 'quantity_ton_battery'].values[0]

separator_US = supply_data.get('separator')['US']
supply_bl.loc[supply_bl['Material'] == 'Separator', 'US'] = (
    separator_US.loc[
        separator_US['year'] == year,
        [
            'LFP separator', 'NCA separator',
            'NMC111 separator', 'NMC532 separator', 'NMC622 separator', 'NMC811 separator',
            'SIB separator'
        ]
    ]
    .sum(axis=1, skipna=True)
    .values[0]
)

# Write binder into the reporting template
for country in ['CaMex', 'FTA', 'FEOC', 'nonFTA']:
    df = supply_data.get('binder')[country]
    supply_bl.loc[supply_bl['Material'] == 'Binder', country] = df.loc[df['year'] == year, 'quantity_ton_battery'].values[0]

# Write cell into the reporting template
for cls in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
    try:
        cell_df = supply_data.get('cell')[cls]
    except KeyError:
        continue
    for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']:
        col_name = f'{chem} cell'
        col_ton  = f'{chem}_ton'
        try:
            value = cell_df.loc[cell_df['year'] == year, col_ton].values[0]
            supply_bl.loc[supply_bl['Material'] == col_name, cls] = value
        except (KeyError, IndexError):
            pass

##### Planned projects #####
planned_project = pd.read_excel('data/core_analysis/template.xlsx')
planned_project.rename(columns={'US': 'US_all',
                               'CaMex': 'CaMex_all'}, inplace=True)
planned_project['US_mature'] = pd.NA
planned_project['CaMex_mature'] = pd.NA

rp_name = ['raw_cobalt', 'cobalt_sulfate', 'raw_graphite', 'raw_lithium', 'raw_manganese', 'raw_nickel',
           'lithium_carbonate', 'lithium_hydroxide', 'manganese_sulfate','nickel_sulfate',
           'lfp_cam', 'nickel_cam', 'graphite_aam', 'silicon_aam',
           'separator', 'electrolyte', 'LIBcell',
           'SIBcell', 'SIBcam', 'SIBaam','SIBelectrolyte'
          ]

rp_data_all = {}
rp_data_mature = {}
for name in rp_name:
    rp_data_all[name] = {}
    rp_data_mature[name] = {}
    for country in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
        file_path_all = f'data/core_analysis/planned/all/{name}_{country}.xlsx'
        file_path_mature = f'data/core_analysis/planned/mature/{name}_{country}.xlsx'
        if os.path.exists(file_path_all):
            try:
                rp_data_all[name][country] = pd.read_excel(file_path_all)
            except Exception:
                pass
        if os.path.exists(file_path_mature):
            try:
                rp_data_mature[name][country] = pd.read_excel(file_path_mature)
            except Exception:
                pass

rp_results_all = {}
rp_results_mature = {}
for name in rp_name:
    rp_results_all[name] = {}
    rp_results_mature[name] = {}
    for country in ['US', 'CaMex', 'FTA', 'FEOC', 'nonFTA']:
        df = rp_data_all.get(name, {}).get(country, None)
        if df is not None and 'year' in df.columns:
            df_year = df[df['year'] == year]
            if 'total_production' in df_year.columns and not df_year.empty:
                value = df_year['total_production'].values[0]
            else:
                value = None
            rp_results_all[name][country] = value
        df = rp_data_mature.get(name, {}).get(country, None)
        if df is not None and 'year' in df.columns:
            df_year = df[df['year'] == year]
            if 'total_production' in df_year.columns and not df_year.empty:
                value = df_year['total_production'].values[0]
            else:
                value = None
            rp_results_mature[name][country] = value

# Write planned project into the reporting template
for country in ['US', 'CaMex']:
    for scen, results in [('all', rp_results_all), ('mature', rp_results_mature)]:
        col = f'{country}_{scen}'
        planned_project.loc[planned_project['Material'] == 'Raw cobalt', col] = results.get('raw_cobalt', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Cobalt sulfate', col] = results.get('cobalt_sulfate', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Raw graphite', col] = results.get('raw_graphite', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Raw lithium', col] = results.get('raw_lithium', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Raw manganese', col] = results.get('raw_manganese', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Raw nickel', col] = results.get('raw_nickel', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Lithium carbonate', col] = results.get('lithium_carbonate', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Lithium hydroxide', col] = results.get('lithium_hydroxide', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Manganese sulfate', col] = results.get('manganese_sulfate', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Nickel sulfate', col] = results.get('nickel_sulfate', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'LFP CAM', col] = results.get('lfp_cam', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Graphite AAM', col] = results.get('graphite_aam', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Silicon AAM', col] = results.get('silicon_aam', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'LIB electrolyte', col] = results.get('electrolyte', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'Separator', col] = results.get('separator', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'SIB cell', col] = results.get('SIBcell', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'SIB CAM', col] = results.get('SIBcam', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'SIB AAM', col] = results.get('SIBaam', {}).get(country, None)
        planned_project.loc[planned_project['Material'] == 'SIB electrolyte', col] = results.get('SIBelectrolyte', {}).get(country, None)

# Write nickel CAMs into the reporting template
for scen, data_source in [('all', rp_data_all), ('mature', rp_data_mature)]:
    for country in ['US', 'CaMex']:

        nickel_cam_df = data_source.get('nickel_cam')[country]
        row = nickel_cam_df.loc[nickel_cam_df['year'] == year]

        col_name = f'{country}_{scen}'

        for chem in ['NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
            planned_project.loc[
                planned_project['Material'] == f'{chem} CAM',
                col_name
            ] = row[chem].values[0]

# Write cells into the reporting template
for scen, data_source in [('all', rp_data_all), ('mature', rp_data_mature)]:
    for country in ['US', 'CaMex']:

        cell_df = data_source.get('LIBcell')[country]
        row = cell_df.loc[cell_df['year'] == year]

        col_name = f'{country}_{scen}'

        for chem in ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811']:
            planned_project.loc[
                planned_project['Material'] == f'{chem} cell',
                col_name
            ] = row[f'{chem}_ton'].values[0]

# Quantify supply from early-stage projects
planned_project.fillna(0, inplace=True)
planned_project['US_unmature'] = planned_project['US_all'] - planned_project['US_mature']
planned_project['CaMex_unmature'] = planned_project['CaMex_all'] - planned_project['CaMex_mature']
planned_project.drop(columns=['US_all', 'CaMex_all'], inplace=True)
planned_project = planned_project[['Material', 'Demand',
                                 'US_mature', 'US_unmature',
                                 'CaMex_mature', 'CaMex_unmature',
                                 'FTA', 'FEOC', 'nonFTA',
                                 'US recycling', 'Non-US recycling',
                                 'Gap'
                         ]]

##### Material supply-demand gaps #####
supply_bl.rename(columns={'US': 'US_mature',
                          'CaMex': 'CaMex_mature'}, inplace=True)

result_temp = supply_bl.set_index('Material').add(planned_project.set_index('Material'), fill_value=0).reset_index()
result_temp = result_temp[['Material', 'Demand',
                           'US_mature', 'US_unmature',
                           'CaMex_mature', 'CaMex_unmature',
                           'FTA', 'FEOC', 'nonFTA',
                           'US recycling', 'Non-US recycling',
                           'Gap'
                         ]]

df_final = result.set_index('Material').add(result_temp.set_index('Material'), fill_value=0).reset_index()
df_final = df_final[['Material', 'Demand',
                     'US_mature', 'US_unmature',
                     'CaMex_mature', 'CaMex_unmature',
                     'FTA', 'FEOC', 'nonFTA',
                     'US recycling', 'Non-US recycling',
                     'Gap'
                     ]]

tier1_columns = ['US_mature']
tier2_columns = ['US_unmature', 'US recycling']
tier3_columns = ['CaMex_mature', 'FTA']
tier4_columns = ['CaMex_unmature']
tier5_columns = ['FEOC', 'nonFTA', 'Non-US recycling']

all_tiers = [tier1_columns, tier2_columns, tier3_columns, tier4_columns, tier5_columns]
all_supply_columns = [col for tier in all_tiers for col in tier]

def adjust_supply(row):
    demand = row['Demand']
    total_supply = row[all_supply_columns].sum()

    if total_supply < demand:
        row['Gap'] = demand - total_supply
        return row

    used_supply = 0
    for tier in all_tiers:
        tier_supply = row[tier].sum()
        if used_supply + tier_supply >= demand:
            remaining_demand = demand - used_supply
            if tier_supply > 0:
                scaling_factor = remaining_demand / tier_supply

                for col in tier:
                    row[col] *= scaling_factor
            lower_tiers = all_tiers[all_tiers.index(tier) + 1:]
            for lower_tier in lower_tiers:
                for col in lower_tier:
                    row[col] = 0
            break
        else:
            used_supply += tier_supply

    row['Gap'] = 0
    return row


df_final = df_final.apply(adjust_supply, axis=1)
df_final.drop(columns=['Non-US recycling'], inplace=True)

# Combine NMC variants for reporting
groups = {
    'NMC CAM': ['NMC111 CAM', 'NMC532 CAM', 'NMC622 CAM', 'NMC811 CAM'],
    'NMC cell': ['NMC111 cell', 'NMC532 cell', 'NMC622 cell', 'NMC811 cell']
}

new_rows = []

for new_name, group_materials in groups.items():
    subset = df_final[df_final['Material'].isin(group_materials)]
    if subset.empty:
        continue
    summed = subset.drop(columns=['Material']).sum(numeric_only=True)
    summed['Material'] = new_name
    new_rows.append(summed)

if new_rows:
    new_df = pd.DataFrame(new_rows)
    detailed_to_drop = [m for mats in groups.values() for m in mats]
    df_final = df_final[
        ~df_final['Material'].isin(detailed_to_drop)
    ]
    df_final = pd.concat([df_final, new_df], ignore_index=True)

# Final formatting
custom_order = ['Bauxite', 'Raw cobalt', 'Raw graphite', 'Raw lithium', 'Raw manganese', 'Raw nickel',
                'Aluminum hydroxide', 'Aluminum sulfate', 'Cobalt sulfate', 'Lithium carbonate', 'Lithium hydroxide',
                'Manganese sulfate', 'Nickel sulfate',
                'LFP CAM', 'NCA CAM', 'NMC CAM', 'Graphite AAM', 'Silicon AAM', 'LIB electrolyte',
                'SIB CAM','SIB AAM', 'SIB electrolyte',
                'Separator', 'Binder',
                'LFP cell', 'NCA cell', 'NMC cell','SIB cell'
               ]

df_final = df_final.set_index('Material').loc[custom_order].reset_index()

df_final.loc[df_final['Material'] == 'Raw cobalt', 'Material'] = 'Cobalt'
df_final.loc[df_final['Material'] == 'Raw graphite', 'Material'] = 'Graphite'
df_final.loc[df_final['Material'] == 'Raw lithium', 'Material'] = 'Lithium'
df_final.loc[df_final['Material'] == 'Raw manganese', 'Material'] = 'Manganese'
df_final.loc[df_final['Material'] == 'Raw nickel', 'Material'] = 'Nickel'
df_final.loc[df_final['Material'] == 'SIB AAM', 'Material'] = 'Na-ion AAM'
df_final.loc[df_final['Material'] == 'SIB CAM', 'Material'] = 'Na-ion CAM'
df_final.loc[df_final['Material'] == 'SIB electrolyte', 'Material'] = 'Na-ion electrolyte'
df_final.loc[df_final['Material'] == 'SIB cell', 'Material'] = 'Na-ion cell'
df_final.loc[df_final['Material'] == 'Silicon AAM', 'Material'] = 'Graphite-silicon AAM'
df_final.loc[df_final['Material'] == 'LIB electrolyte', 'Material'] = 'Li-ion electrolyte'

df_final.rename(columns={'US_mature': 'US (existing & advanced-stage projects)',
                         'US_unmature': 'US (early-stage projects)',
                         'CaMex_mature': 'Canada & Mexico (historical average & advanced-stage)',
                         'CaMex_unmature': 'Canada & Mexico (early-stage)',
                         'FTA': 'Other FTA countries (historical average)',
                         'FEOC': 'FEOC countries (historical average)',
                         'nonFTA': 'Other non-FTA countries (historical average)',
                         'US recycling': 'US recycling',
                         'Gap': 'Gap'}, inplace=True)
print(df_final.round(0))
