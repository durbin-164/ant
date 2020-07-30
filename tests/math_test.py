import unittest 
import ant 

class MainTest(unittest.TestCase):
    def test_add(self):
        self.assertEqual(ant.add(10,20), 30)

    def test_subtract(self):
        self.assertEqual(ant.subtract(20,10), 10)

        