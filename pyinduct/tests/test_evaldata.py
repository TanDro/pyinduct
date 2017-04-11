import unittest
import numpy as np
import pyinduct as pi


class EvalDataTest(unittest.TestCase):
    def setUp(self):
        testDataTime1 = pi.core.Domain((0, 10), 11)
        testDataSpatial1 = pi.core.Domain((0, 1), 5)
        testOutputData1 = np.random.rand(11, 5)
        self.data1 = pi.core.EvalData(input_data=[testDataTime1, testDataSpatial1],
                                      output_data=testOutputData1)

        testDataTime2 = pi.core.Domain((0, 10), 11)
        testDataSpatial2 = pi.core.Domain((0, 1), 5)
        testOutputData2 = np.random.rand(11, 5)
        self.data2 = pi.core.EvalData(input_data=[testDataTime2, testDataSpatial2],
                                      output_data=testOutputData2)

        testDataTime4 = pi.core.Domain((0, 10), 101)
        testDataSpatial4 = pi.core.Domain((0, 1), 11)
        testOutputData4 = np.random.rand(101, 11)
        self.data4 = pi.core.EvalData(input_data=[testDataTime4, testDataSpatial4],
                                      output_data=testOutputData4)

        testDataTime5 = pi.core.Domain((0, 10), 101)
        testDataSpatial5 = pi.core.Domain((0, 1), 11)
        testOutputData5 = -np.random.rand(101, 11)
        self.data5 = pi.core.EvalData(input_data=[testDataTime5, testDataSpatial5],
                                      output_data=testOutputData5)

        testDataTime3 = pi.core.Domain((0, 10), 101)
        testOutputData3 = np.random.rand(101)
        self.data3 = pi.core.EvalData(input_data=[testDataTime3],
                                      output_data=testOutputData3)

    # def test_interpolate1d(self):
    #     data = self.data3.interpolate([[2, 7]])
    #     self.assertTrue(data.output_data[0] == self.data3.output_data[20])
    #     self.assertTrue(data.output_data[1] == self.data3.output_data[70])
    #
    # def test_interpolate2d(self):
    #     data = self.data1.interpolate([[2, 5], [0.5]])
    #     self.assertTrue(data.output_data[0, 0] == self.data1.output_data[2, 2])
    #     self.assertTrue(data.output_data[1, 0] == self.data1.output_data[5, 2])
    #
    # def test_call1d(self):
    #     data = self.data3([[slice(None)]])
    #     self.assertTrue((data.output_data[:] == self.data3.output_data[:]).all())
    #
    # def test_call2d(self):
    #     data = self.data1([[slice(None)], [0.75]])
    #     self.assertTrue((data.output_data[:, 0] == self.data1.output_data[:, 3]).all())
    #
    # def test_call2d_multi(self):
    #     data = self.data1([[slice(None)], [0.25, 0.75]])
    #     self.assertTrue((data.output_data[:, 0] == self.data1.output_data[:, 1]).all())
    #     self.assertTrue((data.output_data[:, 1] == self.data1.output_data[:, 3]).all())
    #
    # def test_call2d_slice(self):
    #     data = self.data1([[slice(1, 5)], [0.75]])
    #     self.assertTrue((data.output_data[:, 0] == self.data1.output_data[1:5, 3]).all())

    # def test_add_const(self):
    #     data = self.data1 + 4
    #     self.assertTrue((data.output_data == self.data1.output_data + 4).all())

    # def test_add_evaldata(self):
    #     data = self.data1 + self.data2
    #     self.assertTrue((data.output_data == self.data1.output_data + self.data2.output_data).all())
    #
    # def test_add_evaldata_diff(self):
    #     data = self.data1 + self.data4
    #     data4red = self.data4(self.data1.input_data)
    #
    #     self.assertTrue((data.output_data == self.data1.output_data + data4red.output_data).all())

    def test_sqrt_evaldata(self):
        data = self.data1.sqrt()

        self.assertTrue((data.output_data == np.sqrt(self.data1.output_data)).all())

    def test_abs_evaldata(self):
        data = self.data5.abs()

        self.assertTrue((data.output_data == np.abs(self.data5.output_data)).all())
