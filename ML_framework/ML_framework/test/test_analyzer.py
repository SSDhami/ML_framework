
import pytest
import pandas as pd
from ML_framework import Analyzer

@pytest.fixture
def analyzer_instance():
    # Create an instance of the Analyzer class for testing
    analyzer = Analyzer('sample.csv')
    return analyzer

def test_read_dataset_csv(analyzer_instance):
    # Test reading dataset from CSV file
    analyzer_instance.read_dataset()
    assert isinstance(analyzer_instance.data_frame, pd.DataFrame)

def test_drop_missing_data(analyzer_instance):
    # Test dropping missing data
    analyzer_instance.data_frame = pd.DataFrame({'A': [1, 2, 3, None], 'B': ['a', 'b', None, 'd']})
    analyzer_instance.drop_missing_data()
    assert len(analyzer_instance.data_frame) == 2

# Add more test functions for other methods in the Analyzer class...

if __name__ == "__main__":
    pytest.main()











