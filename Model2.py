#Importing Libraries
import pyomo.environ as pyo
from pyomo.environ import Var, NonNegativeIntegers,NonNegativeReals,Binary,Objective,minimize,value, DataPortal , PercentFraction
from pyomo.opt import SolverFactory




# Step 1: Initialize the model
model = pyo.ConcreteModel()

# Step 2: Define sets and parameters
model.T = pyo.RangeSet()  # Time periods
model.W = pyo.RangeSet()  # Types of medical waste
model.S = pyo.RangeSet()  # Service providers

# Assuming there's a cost associated with each waste type and service provider
# Define these costs as parameters
model.cost_S1 = pyo.Param(model.W, within=NonNegativeReals)
model.cost_S2 = pyo.Param(model.W, within=NonNegativeReals)

# Assuming there's a limit on how much each service provider can handle per time period
model.limit_S1 = pyo.Param(model.T, within=NonNegativeReals)
model.limit_S2 = pyo.Param(model.T, within=NonNegativeReals)


# Sample parameter setup
model.D = pyo.Param(model.T, model.W, within=NonNegativeReals)  # Demand for medical waste type w in period t
model.C_S1 = pyo.Param(model.T, model.W, within=NonNegativeReals)  # Cost of disposing with ordinary service
model.C_S2 = pyo.Param(model.T, model.W, within=NonNegativeReals)  # Cost of disposing with emergency service
model.Cap_T = pyo.Param(model.T, within=NonNegativeReals)
model.mna = pyo.Param(within=NonNegativeIntegers)
model.Ut_W = pyo.Param(model.W, within=NonNegativeReals)
model.V_T = pyo.Param(model.T, within=NonNegativeReals)
model.U_S = pyo.Param(within=NonNegativeReals)
model.SR_T = pyo.Param(model.T, within=NonNegativeReals)
model.PS_T = pyo.Param(model.T, within=PercentFraction)
model.RC_WST = pyo.Param(model.W, model.T, within=NonNegativeReals)
model.OS_W = pyo.Param(model.W, within=NonNegativeReals)
model.BP_WT = pyo.Param(model.W, model.T, within=NonNegativeReals)
model.GE_T = pyo.Param(model.T, within=NonNegativeReals)
model.OC_WST = pyo.Param(model.W, model.T, within=NonNegativeReals)
model.F_S = pyo.Param(within=NonNegativeReals)
model.CT_W = pyo.Param(model.W, within=NonNegativeReals)
model.RO_W = pyo.Param(model.W, within=NonNegativeReals)
model.M = pyo.Param(within=NonNegativeReals)

# The number of waste w disposed by service provider S in period t
model.X_WTS = pyo.Var(model.W, model.T, bounds=(0.0, None))

# The number of waste w processed in period t
model.MP_WT = pyo.Var(model.W, model.T, bounds=(0.0, None))

# Inventory level of waste w after satisfying processing demands in period t
model.I_WT = pyo.Var(model.W, model.T, within=NonNegativeReals)

# Binary variable: 1 if waste w is processed by service provider S in period t, otherwise 0
model.y_WTS = pyo.Var(model.W, model.T, within=Binary)

# Binary variable: 1 if the beginning inventory level of waste w is less than the reorder point, otherwise 0
model.SE_WT = pyo.Var(model.W, model.T, within=Binary)

# Number of lots of waste w processed by ordinary service provider in period t
model.MO_WT = pyo.Var(model.W, model.T, bounds=(0.0, None))

# Number of lots of waste w processed by emergency service provider in period t
model.ME_WT = pyo.Var(model.W, model.T, bounds=(0.0, None))

# Cost to store a unit of waste w in period t
model.H_WT = pyo.Var(model.W, model.T, within=NonNegativeReals)

# Increase in human resource capacity in hours in period t
model.W_T = pyo.Var(model.T, bounds=(0.0, None))

# Backorder of waste w in period t
model.B_WT = pyo.Var(model.W, model.T, within=NonNegativeReals)

# Percentage of waste w that need to be stored in period t for next period
model.DN_WT = pyo.Var(model.W, model.T, within=PercentFraction)


#Objectives

def objective_cost_minimization(model):
    disposal_costs_S1 = sum(model.C_S1_WT[w, t] * model.MO_WT[w, t] for t in model.T for w in model.W)
    disposal_costs_S2 = sum(model.C_S2_WT[w, t] * model.ME_WT[w, t] for t in model.T for w in model.W)
    storage_costs = sum(model.I_WT[w, t] * model.H_WT[w, t] for t in model.T for w in model.W)
    hr_capacity_increase_costs = sum(model.W_T[t] * model.V_T[t] for t in model.T)
    backorder_costs = sum(model.BP_WT[w, t] * model.B_WT[w, t] for t in model.T for w in model.W)
    transportation_costs = sum(model.X_WTS[w, t, s] * model.F_S[s] * model.CT_W[w] for s in model.S for t in model.T for w in model.W)
    fixed_processing_costs = sum(model.OC_WST[w, s, t] * model.SE_WT[w, t] for s in model.S for t in model.T for w in model.W)

    return (disposal_costs_S1 + disposal_costs_S2 + storage_costs + hr_capacity_increase_costs + 
            backorder_costs + transportation_costs + fixed_processing_costs)

model.cost_minimization_objective = Objective(rule=objective_cost_minimization, sense=minimize)


# Define the second objective function to maximize utility
def objective_utility_maximization(model):
    utility_sum = sum(model.U_S[s] * model.y_WTS[w, t, s] for s in model.S for t in model.T for w in model.W)
    return utility_sum


#Constraints

# Define the inventory balance constraint
def inventory_balance_constraint(model, w, t):
    if t == model.T.first():  # No previous period for the first time period
        return model.I_WT[w, t] == model.MO_WT[w, t] + model.ME_WT[w, t] - model.MP_WT[w, t] - model.B_WT[w, t]
    else:
        return model.I_WT[w, t] == model.I_WT[w, t-1] + model.MO_WT[w, t] + model.ME_WT[w, t] - model.MP_WT[w, t] - model.B_WT[w, t-1] + model.B_WT[w, t]
# Add the inventory balance constraint to the model
model.inventory_balance_constraint = pyo.Constraint(model.W, model.T, rule=inventory_balance_constraint)



# Define the waste processing limit constraint
def waste_processing_limit_constraint(model, w, s, t):
    return model.MO_WT[w, t] + model.ME_WT[w, t] <= model.Y_WST[w, s, t] * model.M
# Add the waste processing limit constraint to the model
model.waste_processing_limit_constraint = pyo.Constraint(model.W, model.S, model.T, rule=waste_processing_limit_constraint)


# Define the constraint
def processing_constraint(model, w, t):
    return model.MP_WT[w, t] <= model.MO_WT[w, t] + model.ME_WT[w, t] + model.I_WT[w, t-1] - model.B_WT[w, t-1]
# Add the constraint to the model
model.processing_constraint = pyo.Constraint(model.W, model.T, rule=processing_constraint)



# Define the constraint
def inventory_constraint(model, w, t):
    return model.I_WT[w, t-1] - model.RO_W[w] + 1 <= model.M * (1 - model.SE_WT[w, t])
# Add the constraint to the model
model.inventory_constraint = pyo.Constraint(model.W, model.T, rule=inventory_constraint)


# Define the constraint
def reorder_constraint(model, w, t):
    return model.RO_W[w] - model.I_WT[w, t-1] <= model.M * model.SE_WT[w, t]
# Add the constraint to the model
model.reorder_constraint = pyo.Constraint(model.W, model.T, rule=reorder_constraint)


# Define the constraint
def balance_constraint(model, t):
    return sum(model.X_WTS[w, t, s] for s in model.S for w in model.W) - sum(model.MP_WT[w, t] for w in model.W) == sum(model.I_WT[w, t] for w in model.W)
# Add the constraint to the model
model.balance_constraint = pyo.Constraint(model.T, rule=balance_constraint)


# Define the constraint
def capacity_constraint(model, t):
    return sum(model.Ut_W[w] * model.X_WTS[w, t, s] for s in model.S for w in model.W) <= model.Cap_T[t] + model.W_T[t]
# Add the constraint to the model
model.capacity_constraint = pyo.Constraint(model.T, rule=capacity_constraint)


# Define the constraint
def constraint_sum_Y_WST(model, t):
    return sum(model.y_WTS[w, t, s] for s in model.S for w in model.W) <= model.mna
# Add the constraint to the model
model.sum_Y_WST_constraint = pyo.Constraint(model.T, rule=constraint_sum_Y_WST)


# Define the constraint
def constraint_sum_X_WTS(model, w, t):
    return model.MO_WT[w, t] + model.ME_WT[w, t] == sum(model.X_WTS[w, t, s] for s in model.S)
# Add the constraint to the model
model.sum_X_WTS_constraint = pyo.Constraint(model.W, model.T, rule=constraint_sum_X_WTS)


# Define the constraint
def constraint_sum_X_I_OS(model, t):
    return sum((model.X_WTS[w, t, s] + model.I_WT[w, t-1]) * model.OS_W[w] for s in model.S for w in model.W) <= model.SR_T[t]
# Add the constraint to the model
model.sum_X_I_OS_constraint = pyo.Constraint(model.T, rule=constraint_sum_X_I_OS)


# Define the constraint
def constraint_D_MP_B(model, t, w):
    return model.DN_WT[w, t] * model.MP_WT[w, t] == model.B_WT[w, t]

# Add the constraint to the model
model.D_MP_B_constraint = pyo.Constraint(model.T, model.W, rule=constraint_D_MP_B)


# Define the constraint
def constraint_MO_RC(model, t, w):
    return sum(model.MO_WT[w, t] for s in model.S) <= sum(model.RC_WST[w, s, t] for s in model.S)

# Add the constraint to the model
model.MO_RC_constraint = pyo.Constraint(model.T, model.W, rule=constraint_MO_RC)


# Define the constraint
def constraint_RO_MO(model, t, w):
    return model.RO_W[w] <= model.MO_WT[w, t]
# Add the constraint to the model
model.RO_MO_constraint = pyo.Constraint(model.T, model.W, rule=constraint_RO_MO)



# Constraint [14]
def constraint_WT_PS_Cap(model, t):
    return model.W_T[t] <= model.PS_T[t] * model.Cap_T[t]

model.WT_PS_Cap_constraint = pyo.Constraint(model.T, rule=constraint_WT_PS_Cap)

# Constraint [15]
def constraint_WT_GE(model, t):
    return model.W_T[t] <= model.GE_T[t]

model.WT_GE_constraint = pyo.Constraint(model.T, rule=constraint_WT_GE)

# Constraint [16]
def constraint_I_DN_MO(model, t, w):
    return model.I_WT[w, t] >= model.DN_WT[w, t] * model.MO_WT[w, t-1]

model.I_DN_MO_constraint = pyo.Constraint(model.T, model.W, rule=constraint_I_DN_MO)

# Constraint [17]
def constraint_MO_D(model, t, w):
    return model.MO_WT[w, t] <= model.D_WT[t]

model.MO_D_constraint = pyo.Constraint(model.T, model.W, rule=constraint_MO_D)

# Constraint [18]
def constraint_X_RC_nonneg(model, t, w, s):
    return model.X_WTS[w, t, s] * model.RC_WST[w, s, t] >= 0

model.X_RC_nonneg_constraint = pyo.Constraint(model.T, model.W, model.S, rule=constraint_X_RC_nonneg)

# Constraint [19]
def constraint_vars_nonneg(model, t, w):
    return model.I_WT[w, t] >= 0 and model.MO_WT[w, t] >= 0 and model.ME_WT[w, t] >= 0 and model.MP_WT[w, t] >= 0 and model.B_WT[w, t] >= 0 and model.DN_WT[w, t] >= 0 and model.D_WT[t] >= 0

model.vars_nonneg_constraint = pyo.Constraint(model.T, model.W, rule=constraint_vars_nonneg)

# Constraint [20]
def constraint_RO_Ut_nonneg(model, w):
    return model.RO_W[w] * model.Ut_W[w] >= 0

model.RO_Ut_nonneg_constraint = pyo.Constraint(model.W, rule=constraint_RO_Ut_nonneg)

# Constraint [21]
def constraint_Cap_WT_GE_nonneg(model, t):
    return model.Cap_T[t] * model.W_T[t] * model.GE_T[t] >= 0

model.Cap_WT_GE_nonneg_constraint = pyo.Constraint(model.T, rule=constraint_Cap_WT_GE_nonneg)

# Constraint [22]
def constraint_SE_y_binary(model, t, w):
    return model.SE_WT[w, t] * model.y_WTS[w, t, 'S1'] in {0, 1}

model.SE_y_binary_constraint = pyo.Constraint(model.T, model.W, rule=constraint_SE_y_binary)



# Creating a model instance (combining the Abstract model with a specific data file)
data = DataPortal()  # A DataPortal() object knows how to read data files
data.load(filename="cslAbstractData.dat", model=model)  # Load the data file information into memory
instance = model.create_instance(data)  # Join the model information and the data file info to create a linear program instance
print(instance.is_constructed())
instance.pprint()

# Solving the instance
optimizer = SolverFactory("glpk")
optimizer.solve(instance)

# Display key values and results

# Display decision variable values
print("\nDecision Variables:")
for w in instance.W:
    for t in instance.T:
        for s in instance.S:
            print(f"X_WTS[{w},{t},{s}] =", instance.X_WTS[w, t, s].value)
            print(f"y_WTS[{w},{t},{s}] =", instance.y_WTS[w, t, s].value)

# Display objective function values
print("\nObjective Function Values:")
print("Total Cost:", instance.cost_minimization_objective.expr())

# Display constraint violation values
print("\nConstraint Violation Values:")
for constr in instance.component_objects(pyo.Constraint, active=True):
    print(constr.local_name, ":")
    for index in constr:
        print(f"  {index}: {constr[index].body() - constr[index].lhs()}")
