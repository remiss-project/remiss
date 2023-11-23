import unittest

from app import create_app


class AppTestCase(unittest.TestCase):
    def test_main(self):
        app = create_app()
        self.assertIsNotNone(app)


if __name__ == '__main__':
    unittest.main()
