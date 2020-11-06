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

# Files
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