import numpy as np
import pandas as pd
from pathlib import Path

class merge_layers:
    def test_existing_layers(self, merge_layers):
        expected = ['htgwl5', 'gwl_up', 'mi', 'egm', 'pt', 'unstructured_Sediments']

        result = merge_layers.layer_names

        assert all(key in result for key in expected)

    def test_layer_thicknesses(self, merge_layers):
        expected =[50.30341692, 11.79997677, 11.40990532, 10., 11.69394407,
               17.89276308]
        result= merge_layers.layer_stats['thickness'].values
        np.testing.assert_almost_equal(
                    result,
                    expected,
                )
    def test_hydrogeoproperties (self,merge_layers):
        expected = [[1.00000000e-07, 8.36000000e-06, 7.00000000e-04, 1.00000000e-07,
                8.36000000e-06, 7.00000000e-04, 8.00000000e-02, 1.70000000e-01,
                2.60000000e-01],
               [1.00000000e-08, 8.94000000e-07, 8.00000000e-05, 1.00000000e-08,
                8.94000000e-07, 8.00000000e-05, 6.00000000e-02, 1.10000000e-01,
                2.20000000e-01],
               [1.00000000e-04, 4.50000000e-04, 2.00000000e-03, 1.00000000e-04,
                4.50000000e-04, 2.00000000e-03, 2.10000000e-01, 2.60000000e-01,
                3.00000000e-01],
               [5.00000000e-12, 5.47000000e-09, 6.00000000e-06, 5.00000000e-12,
                5.47000000e-09, 6.00000000e-06, 1.00000000e-02, 1.00000000e-01,
                2.00000000e-01],
               [4.44219040e-04, 1.05020286e-03, 2.55780960e-03, 4.43948025e-04,
                1.05010340e-03, 2.55233993e-03, 2.23265712e-01, 2.63265712e-01,
                3.08843808e-01],
               [8.31064956e-09, 1.88208934e-06, 4.73627041e-04, 8.28227555e-09,
                1.85432330e-06, 4.63344532e-04, 1.12970393e-01, 1.83242598e-01,
                2.53514804e-01]
               ]
        result = merge_layers.layer_stats[[col for col in merge_layers.layer_stats if col !='thickness']].values
        
        np.testing.assert_almost_equal(
                    result,
                    expected,
                )




