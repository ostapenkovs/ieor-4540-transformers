# Utility functions to help train, validate, and analyze models and their results.
from torchvision import datasets, transforms

ALLOWED_DATASETS = ('cifar10', 'mnist')

def get_data(dataset_name, train_dir, test_dir, dwn):
    assert dataset_name in ALLOWED_DATASETS, f'dataset_name must be one of {ALLOWED_DATASETS}.'

    if dataset_name == 'cifar10':
        my_dataset = datasets.CIFAR10
        channels, image_size, num_classes = 3, 32, 10
        # transform = [
        #     transforms.RandomCrop(image_size, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomRotation(10)
        # ]
    elif dataset_name == 'mnist':
        my_dataset = datasets.MNIST
        channels, image_size, num_classes = 1, 28, 10
        # transform = [
        #     #
        # ]
    
    transform = [transforms.ToTensor(), transforms.Normalize([0.5]*channels, [0.5]*channels)]
    transform = transforms.Compose(transform)
    train_set = my_dataset(root=train_dir, train=True, download=dwn, transform=transform)
    test_set  = my_dataset(root=test_dir, train=False, download=dwn, transform=transform)

    return train_set, test_set, channels, image_size, num_classes

if __name__ == '__main__':
    pass
