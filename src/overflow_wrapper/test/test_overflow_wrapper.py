
import unittest
import os
from overflow_wrapper.overflow_wrapper import OverflowWrapper


class OverflowWrapperTestCase(unittest.TestCase):

    def setUp(self):
        pass
        
    def tearDown(self):
        
        for filename in ['over.namelist']:
            if os.path.exists(filename):
                os.remove(filename)    
        
    def test_Overflow(self):
        
        dirname = os.path.abspath(os.path.dirname(__file__))

        basename = os.getcwd()
        os.chdir(dirname)

        try:
            
            startfile_name = 'overflow1.inp'
            infile_name = 'overflow1_out.inp'
            #dumpfile = 'overflow1.dump'
            
            comp = OverflowWrapper()
    
            # Check input file generation
    
            comp.load_model(startfile_name)
            comp.overflowD = False
            comp.generate_input()
    
            file1 = open(infile_name, 'r')
            result1 = file1.readlines()
            file1.close()
            file2 = open('over.namelist', 'r')
            result2 = file2.readlines()
            file2.close()
    
            lnum = 1
            for line1, line2 in zip(result1, result2):
                try:
                    self.assertEqual(line1, line2)
                except AssertionError as err:
                    raise AssertionError("line %d doesn't match file %s: %s" % (lnum, 
                                                                                infile_name,
                                                                                str(err)))
                lnum += 1
    
            
        finally:
            os.chdir(basename)
        
if __name__ == "__main__":

    unittest.main()
    
