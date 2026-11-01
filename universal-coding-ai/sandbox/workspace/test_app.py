import unittest
import app

class TestApp(unittest.TestCase):
    def test_module_exists(self):
        self.assertTrue(hasattr(app, '__file__'))

if __name__ == '__main__':
    unittest.main()
