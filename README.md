# bilevel_optimized_INR

Code repo for *Bilevel Optimized Implicit Neural Representation for Scan-Specific Accelerated MRI Reconstruction*.

A preprint is available at **https://arxiv.org/abs/2502.21292**

For peer-review purpose only the project layout and public APIs are shown here; full code will be public upon manuscript acceptance.

## Repo layout

```text
bilevel_optimized_INR/
├── pyproject.toml              # package metadata & deps
├── README.md                  
│
├── inr/                        
│   ├── __init__.py            
│   ├── config.py               # load JSON / commentJSON configs
│   ├── train.py                # INR training
│   ├── model.py                # network setup
│   ├── data.py                 # loaders for k-space & coil maps
│   └── utils/                  
│       ├── __init__.py
│       └── utils.py            # all utility functions 
│
├── bilevel_inr.py         # Bilevel optimized INR script
└── bilevel_mbir.py        # Bilevel optimized MBIR script




