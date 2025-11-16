import cobra
model = cobra.test.create_test_model("textbook")
solution = model.optimize()
solution.objective_value
