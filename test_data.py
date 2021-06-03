import pytest
import random

from torch.tensor import Tensor
with pytest.warns(DeprecationWarning):
    from data_preprocessing import Buildings


class TestDataset:
    dataset = Buildings()
    
    def test_dataset_instantiation(self):
        assert self.dataset
        
    def test_dataset_indexing(self):
        assert self.dataset[0]
        assert self.dataset[1]
        assert self.dataset[len(self.dataset)-1]
        with pytest.raises(Exception):
            self.dataset[len(self.dataset)]
            
    def test_dataset_output(self):
        idx = int(random.random() * len(self.dataset))
        output = self.dataset[idx]
        assert isinstance(output, tuple)
        assert isinstance(output[0], Tensor)
        assert isinstance(output[1], Tensor)
        assert output[0].dim() == 3
        assert output[1].dim() == 2
        assert output[0].size(-1) == output[1].size(-2)
        assert output[1].size(-1) == output[1].size(-2)
        
    def test_color_jitter(self):
        ...
        
    # def test_data_return_slice(self):
    #     assert TestData.dataset[10:15]
    
class TestDataloader:
    ...
