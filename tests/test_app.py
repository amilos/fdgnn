import unittest

class TestApp(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()