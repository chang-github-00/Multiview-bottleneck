from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomAffine, ToTensor, ToPILImage
from utils.data import PixelCorruption, MultiViewDataset
from utils.evaluation_new import evaluate, split
from torchvision.datasets import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import training as training_module

if __name__ == '__main__':
    mnist_dir = '.MNIST'
    train_set = MNIST(mnist_dir, download=True, train=True, transform=ToTensor())
    test_set = MNIST(mnist_dir, download=True, train=False, transform=ToTensor())
    # Defining the augmentations
    t = Compose([
        RandomAffine(degrees=180,
                     translate=[0.1, 0.1],
                     scale=[0.9, 1.1]),  # Small affine transformations
        ToTensor(),              # Conversion to torch tensor
        PixelCorruption(0.8)     # PixelCorruption with keep probability 80%
    ])

    # Creating the multi-view dataset using the augmentation class defined by t
    mv_train_set = MultiViewDataset(MNIST(mnist_dir, download=True), t)

    # Initialization of the data loader
    train_loader = DataLoader(mv_train_set, batch_size=64, shuffle=True, num_workers=8)
    TrainerClass = getattr(training_module, 'MIB_newTrainer')
    trainer = TrainerClass(lr=1e-4)
    train_subset = split(mv_train_set,100,'Random')
    test_set = split(mv_train_set,100,'Random')
    trainer.to('cuda')
    for epoch in tqdm(range(10000)):
        for data in (train_loader):
            trainer.train_step(data)
        train_accuracy, test_accuracy = evaluate(encoder=trainer.encoder_v1,train_on=train_subset,test_on=test_set,device='cuda')
        tqdm.write('Train Accuracy: %f' % train_accuracy)
        tqdm.write('Test Accuracy: %f ' % test_accuracy)