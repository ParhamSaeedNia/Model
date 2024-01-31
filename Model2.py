#-----IMPORTING LIBRARIES-----
import pyomo.environ as pyo
from pyomo.environ import NonNegativeIntegers, NonNegativeReals, Binary, Objective, minimize, maximize, PercentFraction
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np


#-----CREATING THE MODEL-----
model = pyo.ConcreteModel()



#-----DEFINING SETS-----
model.T = pyo.RangeSet(1, 100)        # Time periods
model.W = pyo.RangeSet(1, 5)         # Types of medical waste
model.S = pyo.Set(initialize=[1, 2]) # Service providers



#-----DEFINING PARAMETERS-----

# Defining waste generation parameter as stochastic
def stochastic_D_WT_init(model, t, w):
    mean_value = 5000000
    std_dev = 1000000
    return np.random.normal(loc=mean_value, scale=std_dev)

model.D_WT = pyo.Param(model.T, model.W, within=NonNegativeReals, mutable=True, initialize=stochastic_D_WT_init)


# Cost parameters for service providers
model.Cs1 = pyo.Param(model.T, model.W, model.S, within=NonNegativeReals, mutable=True)
model.Cs2 = pyo.Param(model.T, model.W, model.S, within=NonNegativeReals, mutable=True)


# Human resource capacity in hours in period t
model.Cap_T = pyo.Param(model.T, within=NonNegativeReals, mutable=True)


# Maximum number of waste lots disposed of from different waste types
model.mna = pyo.Param(within=NonNegativeIntegers, mutable=True)


# Unitary time for managing a unit of waste w
model.Ut_W = pyo.Param(model.W, within=NonNegativeReals, mutable=True)


# Cost to increase in human resource capacity per hour in period t
model.V_T = pyo.Param(model.T, within=NonNegativeReals, mutable=True)


# Utility of choosing waste disposal service provider S1 or S2
model.U_S = pyo.Param(model.S, within=NonNegativeReals, mutable=True)


# Storeroom available in period t
model.SR_T = pyo.Param(model.T, within=NonNegativeReals, mutable=True)


# Percentage of human resource capacity which can be used in overtime in period t
model.PS_T = pyo.Param(model.T, within=PercentFraction, mutable=True)


# Regular capacity of service provider S1 or S2 for waste w in period t
model.RC_WST = pyo.Param(model.W, model.S, model.T, within=NonNegativeReals, mutable=True)


# Space required for a unit of waste w
model.OS_W = pyo.Param(model.W, within=NonNegativeReals, mutable=True)


# Unit backorder penalty of waste w in period t
model.BP_WT = pyo.Param(model.W, model.T, within=NonNegativeReals, mutable=True)


# Maximum available overtime in hours in period t
model.GE_T = pyo.Param(model.T, within=NonNegativeReals, mutable=True)


# Fixed processing cost of waste w in period t from service provider S1 or S2
model.OC_WST = pyo.Param(model.W, model.S, model.T, within=NonNegativeReals, mutable=True)


# Distance between the hospital and waste disposal service S1 or S2 in kilometer
model.F_S = pyo.Param(model.S, within=NonNegativeReals, mutable=True)


# Cost of transporting one unit of waste w per kilometer
model.CT_W = pyo.Param(model.W, within=NonNegativeReals, mutable=True)


# Reorder point for waste w
model.RO_W = pyo.Param(model.W, within=NonNegativeReals, mutable=True)


# Percentage of waste w that need to be stored in period t for next period
model.DN_WT = pyo.Param(model.W, model.T, within=PercentFraction, mutable=True)


# Cost to store a unit of waste w in period t
model.H_WT = pyo.Param(model.T, model.W, within=NonNegativeReals, mutable=True)


# A big number
model.M = pyo.Param(within=NonNegativeReals, mutable=True)


#-----SAMPLE PARAMETER SETUP-----

# Set cost values for service provider 1 (S1) for each waste type
model.Cs1_indexed_values = {(t, w, s): 5000 for t in model.T for w in model.W for s in model.S if s == 1}  # Adjusted for realism
model.Cs1.store_values(model.Cs1_indexed_values)

# Set cost values for emergency service provider 2 (S2) for each waste type
model.Cs2_indexed_values = {(t, w, s): 10000 for t in model.T for w in model.W for s in model.S if s == 2}  # Adjusted for realism
model.Cs2.store_values(model.Cs2_indexed_values)

# Human resource capacity in hours in period t
model.Cap_T_indexed_values = {t: 2992 for t in model.T}
model.Cap_T.store_values(model.Cap_T_indexed_values)

# Maximum number of waste lots disposed of from different waste types
model.mna_indexed_values = {None: 10}
model.mna.store_values(model.mna_indexed_values)

# Unitary time for managing a unit of waste w
model.Ut_W_indexed_values = {w: 30 for w in model.W}
model.Ut_W.store_values(model.Ut_W_indexed_values)

# Cost to increase in human resource capacity per hour in period t
model.V_T_indexed_values = {t: 10000 for t in model.T}
model.V_T.store_values(model.V_T_indexed_values)

# Utility of choosing waste disposal service provider S1 or S2
model.U_S_indexed_values = {s: 0.8 for s in model.S}
model.U_S.store_values(model.U_S_indexed_values)

# Storeroom available in period t
model.SR_T_indexed_values = {t: 20000 for t in model.T}
model.SR_T.store_values(model.SR_T_indexed_values)

# Percentage of human resource capacity which can be used in overtime in period t
model.PS_T_indexed_values = {t: 0.003 for t in model.T}
model.PS_T.store_values(model.PS_T_indexed_values)

# Regular capacity of service provider S1 or S2 for waste w in period t
model.RC_WST_indexed_values = {(w, s, t): 100 for w in model.W for s in model.S for t in model.T}
model.RC_WST.store_values(model.RC_WST_indexed_values)

# Space required for a unit of waste w
model.OS_W_indexed_values = {w: 100 for w in model.W}
model.OS_W.store_values(model.OS_W_indexed_values)

# Unit backorder penalty of waste w in period t
model.BP_WT_indexed_values = {(w, t): 500 for w in model.W for t in model.T}
model.BP_WT.store_values(model.BP_WT_indexed_values)

# Maximum available overtime in hours in period t
model.GE_T_indexed_values = {t: 8 for t in model.T}
model.GE_T.store_values(model.GE_T_indexed_values)

# Fixed processing cost of waste w in period t from service provider S1 or S2
model.OC_WST_indexed_values = {(w, s, t): 500 for w in model.W for s in model.S for t in model.T}
model.OC_WST.store_values(model.OC_WST_indexed_values)

# Distance between the hospital and waste disposal service S1 or S2 in kilometer
model.F_S_indexed_values = {s: 35 for s in model.S}
model.F_S.store_values(model.F_S_indexed_values)

# Cost of transporting one unit of waste w per kilometer
model.CT_W_indexed_values = {w: 5000 for w in model.W}
model.CT_W.store_values(model.CT_W_indexed_values)

# Reorder point for waste w
model.RO_W_indexed_values = {w: 10 for w in model.W}
model.RO_W.store_values(model.RO_W_indexed_values)

# Cost to store a unit of waste w in period t
model.H_WT_indexed_values = {(t, w): 1000 for t in model.T for w in model.W}
model.H_WT.store_values(model.H_WT_indexed_values)

# Percentage of waste w that need to be stored in period t for next period
model.DN_WT_indexed_values = {(w, t): 0.3 for w in model.W for t in model.T}
model.DN_WT.store_values(model.DN_WT_indexed_values)

# A big number
model.M_indexed_values = {None: 100000}
model.M.store_values(model.M_indexed_values)


#-----DEFINING VARIABLES-----

# The number of waste w disposed by service provider S1 or S2 in period t
model.X_WTS = pyo.Var(model.W, model.T, model.S, domain=NonNegativeReals)

# The number of waste w acquired in period t
model.MP_WT = pyo.Var(model.W, model.T, domain=NonNegativeReals)

# Inventory level of waste w after satisfying processing demands in period t
model.I_WT = pyo.Var(model.W, model.T, domain=NonNegativeReals)

# Binary variable: 1 if waste w is processed by service provider S in period t, otherwise 0
model.Y_WST = pyo.Var(model.W, model.S, model.T, within=Binary)

# Binary variable: 1 if the beginning inventory level of waste w is less than the reorder point, otherwise 0
model.SE_WT = pyo.Var(model.W, model.T, within=Binary)

# Number of lots of waste w processed by ordinary service provider in period t
model.MO_WT = pyo.Var(model.W, model.T, domain=NonNegativeReals)

# Number of lots of waste w processed by emergency service provider in period t
model.ME_WT = pyo.Var(model.W, model.T, domain=NonNegativeReals)

# Increase in human resource capacity in hours in period t
model.W_T = pyo.Var(model.T, domain=NonNegativeReals)

# Backorder of waste w in period t
model.B_WT = pyo.Var(model.W, model.T, domain=NonNegativeReals)




#-----DEFINING OBJECTIVE FUNCTION-----

#-----OBJECTIVE [1]-----

# Define cost components
def objective_cost_minimization(model):
    # Disposal costs for service provider S1
    disposal_costs_S1 = sum(model.Cs1[t, w, 1] * model.MO_WT[w, t] for t in model.T for w in model.W)
    
    # Disposal costs for emergency service provider S2
    disposal_costs_S2 = sum(model.Cs2[t, w, 2] * model.ME_WT[w, t] for t in model.T for w in model.W)
    
    # Storage costs
    storage_costs = sum(model.I_WT[w, t] * model.H_WT[t, w] for t in model.T for w in model.W)
    
    # Human resource capacity increase costs
    hr_capacity_increase_costs = sum(model.W_T[t] * model.V_T[t] for t in model.T)
    
    # Backorder costs
    backorder_costs = sum(model.BP_WT[w, t] * model.B_WT[w, t] for t in model.T for w in model.W)
    
    # Transportation costs
    transportation_costs = sum(model.X_WTS[w, t, s] * model.F_S[s] * model.CT_W[w] for s in model.S for t in model.T for w in model.W)
    
    # Fixed processing costs
    fixed_processing_costs = sum(model.OC_WST[w, s, t] * model.SE_WT[w, t] for s in model.S for t in model.T for w in model.W)

    # Combine cost components into the objective function
    return (disposal_costs_S1 + disposal_costs_S2 + storage_costs + hr_capacity_increase_costs + 
            backorder_costs + transportation_costs + fixed_processing_costs)

# Set up the objective function for minimization
model.cost_minimization_objective = Objective(rule=objective_cost_minimization, sense=minimize)



# #-----OBJECTIVE [2]-----

# # Define the second objective function to maximize utility
# def objective_utility_maximization(model):
#     utility_sum = sum(model.U_S[s] * model.Y_WST[w, s, t] for s in model.S for t in model.T for w in model.W)
#     return utility_sum

# # Add the second objective function to the model
# model.utility_maximization_objective = Objective(rule=objective_utility_maximization, sense=maximize)



#-----CONSTRAINTS-----


# Constraint [1]: Inventory Balance Constraint
def inventory_balance_constraint(model, w, t):
    if t == model.T.first():  # No previous period for the first time period
        return model.I_WT[w, t] == model.MO_WT[w, t] + model.ME_WT[w, t] - model.MP_WT[w, t] - model.B_WT[w, t]
    else:
        return model.I_WT[w, t] == model.I_WT[w, t-1] + model.MO_WT[w, t] + model.ME_WT[w, t] - model.MP_WT[w, t] - model.B_WT[w, t-1] + model.B_WT[w, t]

# Add the inventory balance constraint to the model
model.inventory_balance_constraint = pyo.Constraint(model.W, model.T, rule=inventory_balance_constraint)


# Constraint [2]: Waste Processing Limit Constraint
def waste_processing_limit_constraint(model, w, s, t):
    return model.MO_WT[w, t] + model.ME_WT[w, t] <= model.Y_WST[w, s, t] * model.M

# Add the waste processing limit constraint to the model
model.waste_processing_limit_constraint = pyo.Constraint(model.W, model.S, model.T, rule=waste_processing_limit_constraint)


# Constraint [3]: Processing Constraint
def processing_constraint(model, w, t):
    if t == model.T.first():
        return model.MP_WT[w, t] <= model.MO_WT[w, t] + model.ME_WT[w, t] + model.OS_W[w]
    else:
        return model.MP_WT[w, t] <= model.MO_WT[w, t] + model.ME_WT[w, t] + model.I_WT[w, t-1] - model.B_WT[w, t-1]

# Add the Processing Constraintt to the model
model.processing_constraint = pyo.Constraint(model.W, model.T, rule=processing_constraint)


# Constraint [4]: Inventory Constraint
def inventory_constraint(model, w, t):
    if t == model.T.first():
        return pyo.Constraint.Skip  # Skip the constraint for the first time period
    else:
        return model.I_WT[w, t-1] - model.RO_W[w] + 1 <= model.M * (1 - model.SE_WT[w, t])

# Add the Inventory Constraint to the model
model.inventory_constraint = pyo.Constraint(model.W, model.T, rule=inventory_constraint)


# Constraint [5]: Reorder Point Constraint
def reorder_constraint(model, w, t):
    if t == model.T.first():
        return model.RO_W[w] <= model.I_WT[w, t]
    else:
        return model.RO_W[w] - model.I_WT[w, t-1] <= model.M * model.SE_WT[w, t]

# Add the Reorder Point constraint to the model
model.reorder_constraint = pyo.Constraint(model.W, model.T, rule=reorder_constraint)


# Constraint [6]: Balance Constraint
def balance_constraint(model, t):
    return sum(model.X_WTS[w, t, s] for s in model.S for w in model.W) - sum(model.MP_WT[w, t] for w in model.W) == sum(model.I_WT[w, t] for w in model.W)

# Add the Balance constraint to the model
model.balance_constraint = pyo.Constraint(model.T, rule=balance_constraint)


# Constraint [7]: Capacity Constraint
def capacity_constraint(model, t):
    return sum(model.Ut_W[w] * model.X_WTS[w, t, s] for s in model.S for w in model.W) <= model.Cap_T[t] + model.W_T[t]

# Add the constraint to the model
model.capacity_constraint = pyo.Constraint(model.T, rule=capacity_constraint)


# Constraint [8]: Constraint on the sum of Y_WST
def constraint_sum_Y_WST(model, t):
    return sum(model.Y_WST[w, s, t] for s in model.S for w in model.W) <= model.mna

# Add the constraint to the model
model.sum_Y_WST_constraint = pyo.Constraint(model.T, rule=constraint_sum_Y_WST)


# Constraint [9]: Constraint on the sum of MO_WT and ME_WT
def constraint_sum_X_WTS(model, t):
    return sum(model.MO_WT[w, t] + model.ME_WT[w, t] for w in model.W) <= sum(model.X_WTS[w, t, s] for s in model.S for w in model.W)

# Add the constraint to the model
model.sum_X_WTS_constraint = pyo.Constraint(model.T, rule=constraint_sum_X_WTS)


# Constraint [10]: Constraint on the sum of X_WTS and I_WT
def constraint_sum_X_I_OS(model, t):
    if t == model.T.first():
        return sum(model.X_WTS[w, t, s] * model.OS_W[w] for s in model.S for w in model.W) <= model.SR_T[t]
    else:
        return sum((model.X_WTS[w, t, s] + model.I_WT[w, t-1]) * model.OS_W[w] for s in model.S for w in model.W) <= model.SR_T[t]

# Add the constraint to the model
model.sum_X_I_OS_constraint = pyo.Constraint(model.T, rule=constraint_sum_X_I_OS)


# Constraint [11]: Constraint on the difference between DN_WT, MP_WT, and B_WT
def constraint_D_MP_B(model, t, w):
    return model.DN_WT[w, t] - model.MP_WT[w, t] <= model.B_WT[w, t]

# Add the constraint to the model
model.D_MP_B_constraint = pyo.Constraint(model.T, model.W, rule=constraint_D_MP_B)


# Constraint [12]: Constraint on the sum of MO_WT and RC_WST
def constraint_MO_RC(model, w, t):
    return sum(model.MO_WT[w, t] for s in model.S) <= sum(model.RC_WST[w, s, t] for s in model.S)

# Add the constraint to the model
model.MO_RC_constraint = pyo.Constraint(model.W, model.T, rule=constraint_MO_RC)


# Constraint [13]: Constraint on the relationship between RO_W and MO_WT
def constraint_RO_MO(model, t, w):
    return model.RO_W[w] <= model.MO_WT[w, t]

# Add the constraint to the model
model.RO_MO_constraint = pyo.Constraint(model.T, model.W, rule=constraint_RO_MO)


# Constraint [14]: Constraint on the relationship between WT_PS_Cap and W_T
def constraint_WT_PS_Cap(model, t):
    return model.W_T[t] <= model.PS_T[t] * model.Cap_T[t]

# Add the constraint to the model
model.WT_PS_Cap_constraint = pyo.Constraint(model.T, rule=constraint_WT_PS_Cap)


# Constraint [15]: Constraint on the relationship between WT_GE and W_T
def constraint_WT_GE(model, t):
    return model.W_T[t] <= model.GE_T[t]

# Add the constraint to the model
model.WT_GE_constraint = pyo.Constraint(model.T, rule=constraint_WT_GE)


# Constraint [16]: Constraint on the relationship between I_WT, DN_WT, and MO_WT
def constraint_I_DN_MO(model, t, w):
    if t == model.T.first():
        return model.I_WT[w, t] <= model.DN_WT[w, t] * model.OS_W[w]
    else:
        return model.I_WT[w, t] <= model.DN_WT[w, t] * model.MO_WT[w, t-1]

# Add the constraint to the model
model.I_DN_MO_constraint = pyo.Constraint(model.T, model.W, rule=constraint_I_DN_MO)


# Constraint [17]: Constraint on the relationship between MO_WT and D_WT
def constraint_MO_D(model, t, w):
    return model.MO_WT[w, t] <= model.D_WT[t,w]

# Add the constraint to the model
model.MO_D_constraint = pyo.Constraint(model.T, model.W, rule=constraint_MO_D)


#-----INSTANCE-----

instance = model.create_instance()  # Join the model information and the data file info to create a linear program instance
print(instance.is_constructed())
instance.pprint()

# Solving the instance
optimizer = SolverFactory("glpk")
result = optimizer.solve(instance)

# Check the solver result status
print(result)

# If the solver result is successful, display the decision variable values and objective functions
if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    # Display decision variable values
    print("\nDecision Variables:")
    for w in instance.W:
        for t in instance.T:
            for s in instance.S:
                print(f"X_WTS[{w},{t},{s}] =", instance.X_WTS[w, t, s].value)

    for w in instance.W:
        for t in instance.T:
            print(f"ME_WT[{w},{t}] =", instance.ME_WT[w, t].value)
            print(f"MO_WT[{w},{t}] =", instance.MO_WT[w, t].value)
            print(f"MP_WT[{w},{t}] =", instance.MP_WT[w, t].value)
            print(f"W_T[{t}] =", instance.W_T[t].value)


    # Display objective function values
    print("\nObjective Function Values:")
    print("Total Cost:", instance.cost_minimization_objective.expr())
    #print("Total Utility:", instance.utility_maximization_objective.expr())

    # Display constraint violation values
    print("\nConstraint Violation Values:")
    for constr in instance.component_objects(pyo.Constraint, active=True):
        print(constr.local_name, ":")
        for index in constr:
            print(f"  {index}: {constr[index].body() - constr[index].expr()}")
else:
    # If the solver result is not successful, print the termination condition and solver message
    print("Solver did not successfully terminate.")
    print("Termination Condition:", result.solver.termination_condition)
    print("Solver Message:")
    print(result.solver.message)
   
    
# Extracting decision variable values from the solved instance
X_WTS_values = np.zeros((len(instance.W), len(instance.T), len(instance.S)))
MO_WT_values = np.zeros((len(instance.W), len(instance.T)))
MP_WT_values = np.zeros((len(instance.W), len(instance.T)))
ME_WT_values = np.zeros((len(instance.W), len(instance.T)))
W_T_values = np.zeros((len(instance.T)))

for i, w in enumerate(instance.W):
    for j, t in enumerate(instance.T):
        for k, s in enumerate(instance.S):
            X_WTS_values[i, j, k] = instance.X_WTS[w, t, s].value
        MO_WT_values[i, j] = instance.MO_WT[w, t].value
        MP_WT_values[i, j] = instance.MP_WT[w, t].value
        ME_WT_values[i, j] = instance.ME_WT[w, t].value
        W_T_values[j] = instance.W_T[t].value
# Extract objective function values
total_cost_value = instance.cost_minimization_objective.expr()
#total_utility_value = instance.utility_maximization_objective.expr()

# Create DataFrames for decision variables
X_WTS_df = pd.DataFrame(X_WTS_values.reshape(len(instance.W), -1), index=instance.W, columns=[f'T={t}, S={s}' for t in instance.T for s in instance.S])
MO_WT_df = pd.DataFrame(MO_WT_values, index=instance.W, columns=instance.T)
MP_WT_df = pd.DataFrame(MP_WT_values, index=instance.W, columns=instance.T)
ME_WT_df = pd.DataFrame(ME_WT_values, index=instance.W, columns=instance.T)
W_T_df = pd.DataFrame(W_T_values, index=instance.T, columns=['W_T'])


# Styler function for DataFrames
def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


# Create an Excel writer using openpyxl
excel_writer = pd.ExcelWriter('output_results.xlsx', engine='openpyxl')

# # Write DataFrames to different sheets in the Excel file
# X_WTS_df.to_excel(excel_writer, sheet_name='X_WTS')
# MO_WT_df.to_excel(excel_writer, sheet_name='MO_WT')
# MP_WT_df.to_excel(excel_writer, sheet_name='MP_WT')
# ME_WT_df.to_excel(excel_writer, sheet_name='ME_WT')
# W_T_df.to_excel(excel_writer, sheet_name='W_T')

# # Save and close the Excel file
# excel_writer.save()

# Display DataFrames
print("\nDecision Variables:")
print("X_WTS:")
print(X_WTS_df)
print("\nMO_WT:")
print(MO_WT_df)
print("\nMP_WT:")
print(MP_WT_df)
print("\nME_WT:")
print(ME_WT_df)
print("\nW_T:")
print(W_T_df)


# Display objective function values
print("\nObjective Function Values:")
print("Total Cost:", total_cost_value)
#print("Total Utility:", total_utility_value)

