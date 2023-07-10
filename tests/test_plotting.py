import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np






import os
from neat.plotting import get_vmec_boundary, get_vmec_magB, plot_animation3d, update
from scipy.io import netcdf_file


class MyTestCase(unittest.TestCase):
    def test_update(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        line, = ax.plot(data[0], data[1], data[2])

        update(2, data, line)

        expected_data = np.array([[1, 2], [5, 6], [9, 10]])
        np.testing.assert_array_equal(line.get_data_3d(), expected_data)

    def test_read_data(self):
        
        wout_filename = os.path.join(
            os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc"
        )
        net_file = netcdf_file(wout_filename, "r", mmap=False)
        self.assertEqual(net_file.variables["nfp"][()], 3)
        net_file.close()

    def test_read_netcdf_array(self):
        wout_filename = "/home/rodrigo/NEAT/examples/inputs/wout_ARIESCS.nc"

        result = get_vmec_boundary(wout_filename)

        self.assertEqual(result[0][0][0][0], 9.564622227455484)

    def test_get_vmec_magB(self):
        wout_filename = "/home/rodrigo/NEAT/examples/inputs/wout_ARIESCS.nc"

        result = get_vmec_magB(wout_filename)

        self.assertEqual(result[0][0], 5.179016053586765)


if __name__ == "__main__":
    unittest.main()
