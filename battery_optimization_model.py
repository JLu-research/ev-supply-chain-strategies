import pandas as pd
from pyomo.environ import ConcreteModel, Var, Constraint, Objective, NonNegativeReals, maximize, SolverFactory

# Years, battery types, and vehicle types
years = list(range(2025, 2036))
battery_types = ['LFP', 'NCA', 'NMC111', 'NMC532', 'NMC622', 'NMC811', 'SIB']
vehicle_types = ['BEV', 'PHEV']

# Materials
materials = [
    'lithium_carbonate', 'lithium_hydroxide', 'nickel_sulfate',
    'graphite', 'manganese_sulfate', 'aluminum_hydroxide',
    'aluminum_sulfate', 'cobalt_sulfate',
    'cam', 'aam', 'electrolyte', 'separator'
]

results = []

for year in years:
    model = ConcreteModel()

    # Decision variables
    model.battery_prod = Var(vehicle_types, battery_types, within=NonNegativeReals)
    material_vars = {
        mat: Var(vehicle_types, battery_types, within=NonNegativeReals)
        for mat in materials
    }

    for mat, var in material_vars.items():
        setattr(model, f'{mat}_eligible', var)

    # Material requirement constraints
    for mat in materials:
        for chem in battery_types:
            for veh in vehicle_types:
                share = material_share.loc[
                    material_share['material'] == mat,
                    f'{veh}_{chem}'
                ].values[0]
                model.add_component(
                    f'{veh}_{chem}_{mat}_constraint',
                    Constraint(expr=material_vars[mat][veh, chem] == model.battery_prod[veh, chem] * share)
                )

    # Supply constraints
    for mat in materials:
        supply_limit = mineral.loc[mineral['year'] == year, f'{mat}_eligible'].values[0]
        total_use = sum(
            material_vars[mat][veh, chem]
            for veh in vehicle_types
            for chem in battery_types
        )
        model.add_component(
            f'{mat}_supply_constraint',
            Constraint(expr=total_use <= supply_limit)
        )

    # Objective: maximize eligible battery production
    model.objective = Objective(
        expr=sum(model.battery_prod[veh, chem] for veh in vehicle_types for chem in battery_types),
        sense=maximize
    )

    # Solve
    solver = SolverFactory('glpk')
    solver.solve(model)

    # Collect results
    results.append({
        'year': year,
        **{
            f'{veh}_{chem}_production': model.battery_prod[veh, chem].value
            for veh in vehicle_types
            for chem in battery_types
        }
    })

# Output
df = pd.DataFrame(results)

