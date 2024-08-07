import unittest

from preprocess import preprocess_multimodal_dataset_data


class MyTestCase(unittest.TestCase):

    def test_preprocess_multimodal(self):
        preprocess_multimodal_dataset_data('../remiss_data_share', '../multimodal_data', )
        # host='srvinv02.esade.es', port=27017)


if __name__ == '__main__':
    unittest.main()
