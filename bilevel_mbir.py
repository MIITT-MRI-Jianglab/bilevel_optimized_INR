# scripts/bilevel_mbir.py
"""
# Bilevel optimization for MBIR training
"""
from bayes_opt import BayesianOptimization
from utils import MBIR

def objective_function(lambda):
    # Dummy objective function for MBIR
    mbir = MBIR()
    raise NotImplementedError

def main():
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds={"lambda": (1e-8, 1e-1)},
        random_state=0,
    )
    optimizer.maximize(init_points=20, n_iter=40)
    print("Best params:", optimizer.max)

if __name__ == "__main__":
    main()