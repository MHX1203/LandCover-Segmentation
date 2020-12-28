from data.dataset import SatImageDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def build_loader(image_list, mask_list, test_size=0.2, augs=None, preprocess=None, num_workers=1, batch_size=32, \
    pin_memory=True, shuffle=True):
    
    # image_masks = list(zip(image_list, mask_list))

    X_train, X_test, y_train, y_test = train_test_split(image_list, mask_list, test_size=test_size, random_state=0, shuffle=True)


    train_set = SatImageDataset(X_train, y_train, augs=augs, preprocess=preprocess, shuffle=True)
    train_loader = DataLoader(train_set, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory, batch_size=batch_size, drop_last=True)

    val_set = SatImageDataset(X_test, y_test, preprocess=preprocess)

    val_loader = DataLoader(val_set, num_workers=num_workers, shuffle=False, pin_memory=pin_memory, batch_size=batch_size, drop_last=False)

    return train_loader, val_loader

