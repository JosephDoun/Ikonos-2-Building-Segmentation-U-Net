import pytest
import random

from torch.tensor import Tensor
from torch.utils.data import DataLoader
with pytest.warns(DeprecationWarning):
    from data_load import Buildings


class TestDataset:
    training = Buildings()
    validation = Buildings(validation=True)
    
    def test_training_dataset_instantiation(self):
        assert self.training
    
    def test_validation_dataset_instantiation(self):
        assert self.validation
    
    def test_training_dataset_indexing(self):
        assert self.training[0]
        assert self.training[1]
        assert self.training[len(self.training)-1]
        with pytest.raises(Exception):
            self.training[len(self.training)]
    
    def test_validation_dataset_indexing(self):
        assert self.validation[0]
        assert self.validation[1]
        assert self.validation[len(self.validation)-1]
        with pytest.raises(Exception):
            self.validation[len(self.validation)]
    
    def test_training_dataset_output(self):
        idx = int(random.random() * len(self.training))
        output = self.training[idx]
        assert isinstance(output, tuple)
        assert isinstance(output[0], Tensor)
        assert isinstance(output[1], Tensor)
        assert output[0].dim() == 3
        assert output[1].dim() == 2
        assert output[0].size(-1) == output[1].size(-2)
        assert output[1].size(-1) == output[1].size(-2)
        
        data = [self.training[i] for i in range(len(self.training))]
        s = set(data)
        assert len(data) == len(s), "The Dataset outputs duplicates"
        
    def test_validation_dataset_output(self):
        idx = int(random.random() * len(self.validation))
        output = self.validation[idx]
        assert isinstance(output, tuple)
        assert isinstance(output[0], Tensor)
        assert isinstance(output[1], Tensor)
        assert output[0].dim() == 3
        assert output[1].dim() == 2
        assert output[0].size(-1) == output[1].size(-2)
        assert output[1].size(-1) == output[1].size(-2)
        
        data = [self.validation[i] for i in range(len(self.validation))]
        s = set(data)
        assert len(data) == len(s), "The Dataset outputs duplicates"
        
    def test_color_jitter(self):
        ...
        
    # def test_data_return_slice(self):
    #     assert TestData.dataset[10:15]
    
class TestDataloader:
    training_loader = DataLoader(Buildings(),
                                 batch_size=256,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True)
    validation_loader = DataLoader(Buildings(validation=True),
                                   batch_size=256,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=True)
    
    def test_loader_instantiations(self):
        assert self.training_loader
        assert self.validation_loader
        
    def test_loader_throughput(self):
        for batch in self.training_loader:
            assert len
        for batch in self.validation_loader:
            print(len(batch))
            
