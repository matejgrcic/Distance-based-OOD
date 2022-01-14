from torch.utils.data import DataLoader
from data.datasets import LostAndFoundWithDistanceDataset

def load_lost_and_found_with_distance(dataroot, bs=1):
    test_set = LostAndFoundWithDistanceDataset(dataroot, split='test')
    print(f"> Loaded {len(test_set)} test images.")
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, pin_memory=True, num_workers=2)
    return test_loader