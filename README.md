This repository contains the code to reproduce all the numerical experiments from the preprint ``Divide and conquer methods for functions of matrices with banded or hierarchical low-rank structure'' by Alice Cortinovis, Daniel Kressner, and Stefano Massei.

-- Correspondence between scripts and figures/tables in the paper --
- doubleSpeedPoly.m -> Fig. 1 & Fig. 5
- testFractional.m -> Table 1
- testSampling.m -> Table 2
- testToeplitz.m -> Table 3
- testNtD.m -> Table 4
- testDensity.m -> Figure 2
- testComputeDiag.m -> Table 5
- test_lag_parameter.m -> Table 6
- testFermiDirac.m -> Table 7
- testAdaptivity.m -> Fig. 6 & Fig. 7
- testTimeSplitting.m -> Figure 8

-- Dependencies --
The following toolboxes are needed to run (some of) the scripts:
- hm-toolbox https://github.com/numpi/hm-toolbox
- metismex and METIS https://github.com/dgleich/metismex
- rktoolbox http://guettel.com/rktoolbox/
- chebfun https://www.chebfun.org/

