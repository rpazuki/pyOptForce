"""End-to-end validation: E. coli succinate overproduction (canonical OptForce case).

Reproduce the published FORCE sets to validate the implementation. Implement once
stages 1-4 are working.
"""

# import cobra
# from pyoptforce import OptForce
#
# model = cobra.io.load_model("iJO1366")  # or e_coli_core for a fast first pass
# of = OptForce(model, target_reaction="EX_succ_e",
#               biomass_reaction="BIOMASS_Ec_iJO1366_core_53p95M",
#               target_fraction=0.5, solver="gurobi")
# of.compute_flux_ranges()
# of.find_must_sets(max_order=2)
# print(of.find_force_sets(k=3, n_solutions=10).to_dataframe())
