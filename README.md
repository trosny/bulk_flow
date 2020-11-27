# Bulk flow seal code
Solution of bulk-flow equations for annular seal

# Anaconda environment setup
Setting up anaconda or miniconda environment to run test scripts

```
	conda env create --name bulk_flow_env -f env.yml
```

followed by

```
	conda activate bulk_flow_env
```

# Directory contents
```
├── LICENSE.txt
├── README.md
├── docs
│   └── pres06Nov2020.pdf
├── env.yml
├── src
│   ├── mesh.py
│   ├── seal.py
│   └── seal_funcs.py
└── tests
    └── test01
        ├── Kanki01_input.yaml
        └── test01.py
```

\tests
  01 : static, basic test of zeroth-order problem solution
  02 : static, how to obtain converged solution on refined grid at high eccentricity
  03 : static, test leakage rate sensitivity to relaxation factor
  04 : dynamic, basic test of first-order problem solution
  05 : static, incompressible, dtu test rig seal input
  06 : test error handling of missing input file parameters
  07 : dynamic, test dyn. coeff. sensitivity to uv_src_blend
  08 : dynamic, test dyn. coeff. sensitivity to relaxation factor

# Example
navigate to tests/test01

```
   python test01.py
```

The test script runs, computes residuals and some results
to the terminal, and generates *.png images

```
└── tests
    └── test01
        ├── Kanki01_input.yaml
        ├── film_thickness_contour.png
        ├── pressure_contour.png
        ├── streamlines.png
        ├── test01.py
        ├── u_contour.png
        └── v_contour.png
```		

# Built With

# Contributing

# Versioning

# Authors

# License
MIT

# Acknowledgments

# Citation

# TODO