"""Device routing tests run on CPU; they do not claim to test CUDA kernels."""

import unittest
from unittest.mock import patch

from device_utils import device_info, resolve_device


class DeviceTests(unittest.TestCase):
    def test_explicit_cpu(self):
        self.assertEqual(resolve_device('cpu'), 'cpu')
        self.assertEqual(device_info('cpu', 'cpu')['actual'], 'cpu')

    @patch('torch.cuda.is_available', return_value=False)
    def test_auto_without_cuda(self, available):
        self.assertEqual(resolve_device('auto'), 'cpu')

    @patch('torch.cuda.is_available', return_value=False)
    def test_requested_cuda_never_falls_back(self, available):
        for device in ['cuda', 'cuda:0']:
            with self.assertRaisesRegex(ValueError, '无法使用 CUDA'):
                resolve_device(device)

    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.is_available', return_value=True)
    def test_auto_and_explicit_gpu_index(self, *mocks):
        self.assertEqual(resolve_device('auto'), 'cuda:0')
        self.assertEqual(resolve_device('cuda'), 'cuda:0')
        self.assertEqual(resolve_device('cuda:1'), 'cuda:1')
        with self.assertRaisesRegex(ValueError, '不存在'):
            resolve_device('cuda:2')

    def test_unsupported_devices(self):
        for device in ['mps', 'bogus', 'cuda:-1', 'cpu:1']:
            with self.assertRaises(ValueError):
                resolve_device(device)


if __name__ == '__main__':
    unittest.main()
