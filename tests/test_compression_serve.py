import pytest
import torch
from compression_serve import CompressionServe, create_compression_serve

class TestCompressionServe:
    def test_init(self):
        m = CompressionServe(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = CompressionServe(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_compression_serve()
        assert isinstance(m, CompressionServe)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
