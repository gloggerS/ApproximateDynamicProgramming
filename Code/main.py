model = Model()

# 3-D array of binary variables
  x = model.addVars(3, 4, 5, vtype=GRB.BINARY)

  # variables index by tuplelist
  l = tuplelist([(1, 2), (1, 3), (2, 3)])
  y = model.addVars(l, ub=[1, 2, 3])

  model.update()