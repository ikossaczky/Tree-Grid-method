# Tree-Grid-method for solving HJB equation
- Tree-Grid.ipynb -implementation of the (1D) Tree-Grid method with examples (uncertain volatility model, passport option pricing model). Solution is the last time layer.
- Tree-Grid-Full.ipynb -implementation of the (1D) Tree-Grid method with examples (uncertain volatility model, passport option pricing model). Solution is the whole time-space domain.
- Tree-Grid-2D.ipynb -implementation of the 2D Tree-Grid method with examples (two-factor uncertain volatility model). Solution is last time layer (2D space domain).
- Tree-Grid-2D-forPaper-Convergence.ipynb -convergence results of the 2D Tree-Grid method in two-factor uncertain volatility model and Black-Scholes model.
- Tree-Grid-2D-forPaper-Plots.ipynb -plots of approximation of solution of two-factor uncertain volatility model computed with 2D Tree-Grid method.
- TreeGrid_module.py -module with all implementations of 1D, 2D Tree-Grid method and all classes defining models, terminal and boundary conditions.
- ExactBSPricing_module.py -functions for computing exact price of two-asset maximum butterfly spread using R-library fExoticOptions.
- Numerics_module.py -usefull functions e.g. for computing errors in different measures etc.
