import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("~/NEAT/src/neat/")
from unittest.mock import MagicMock

import netCDF4 as netcdf
from neat.plotting import get_vmec_boundary, get_vmec_magB, plot_animation3d


class MyTestCase(unittest.TestCase):
    # def test_update(self):
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')

    #     data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    #     line, = ax.plot(data[0], data[1], data[2])

    #     update(2, data, line)

    #     expected_data = np.array([[1, 2], [5, 6], [9, 10]])
    #     np.testing.assert_array_equal(line.get_data_3d(), expected_data)

    # def test_read_data(self):
    #     # Mocking netcdf.netcdf_file
    #     netcdf.netcdf_file = MagicMock()
    #     mock_net_file = netcdf.netcdf_file.return_value
    #     mock_net_file.variables = {
    #         "ns": MagicMock(return_value=2),
    #         "nfp": MagicMock(return_value=3),
    #         "xn": MagicMock(return_value=[1, 2, 3]),
    #         "xm": MagicMock(return_value=[4, 5, 6]),
    #         "xn_nyq": MagicMock(return_value=7),
    #         "xm_nyq": MagicMock(return_value=8),
    #         "rmnc": MagicMock(return_value=[0.1, 0.2, 0.3]),
    #         "zmns": MagicMock(return_value=[0.4, 0.5, 0.6]),
    #         "bmnc": MagicMock(return_value=[0.7, 0.8, 0.9]),
    #         "lasym__logical__": MagicMock(return_value=0),
    #         "rmns": MagicMock(return_value=[]),
    #         "zmnc": MagicMock(return_value=[]),
    #         "bmns": MagicMock(return_value=[]),
    #     }

    #     wout_filename = "example.wout"
    #     net_file = netcdf.netcdf_file(wout_filename, "r", mmap=False)

    #     self.assertEqual(netcdf.netcdf_file.call_args[0][0], wout_filename)
    #     self.assertEqual(netcdf.netcdf_file.call_args[0][1], "r")
    #     self.assertFalse(netcdf.netcdf_file.call_args[1]["mmap"])

    #     self.assertEqual(net_file.variables["ns"].call_count, 1)
    #     self.assertEqual(net_file.variables["nfp"].call_count, 1)
    #     self.assertEqual(net_file.variables["xn"].call_count, 1)
    #     self.assertEqual(net_file.variables["xm"].call_count, 1)
    #     self.assertEqual(net_file.variables["xn_nyq"].call_count, 1)
    #     self.assertEqual(net_file.variables["xm_nyq"].call_count, 1)
    #     self.assertEqual(net_file.variables["rmnc"].call_count, 1)
    #     self.assertEqual(net_file.variables["zmns"].call_count, 1)
    #     self.assertEqual(net_file.variables["bmnc"].call_count, 1)
    #     self.assertEqual(net_file.variables["lasym__logical__"].call_count, 1)
    #     self.assertEqual(net_file.variables["rmns"].call_count, 0)
    #     self.assertEqual(net_file.variables["zmnc"].call_count, 0)
    #     self.assertEqual(net_file.variables["bmns"].call_count, 0)

    #     self.assertEqual(netcdf.netcdf_file.call_count, 1)
    #     self.assertEqual(len(net_file.variables["xn"].return_value), 3)

    #     net_file.close()

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
