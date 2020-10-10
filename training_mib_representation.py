import os
#os.chdir ('/content/drive/My Drive/Multi-View-Information-Bottleneck')
#os.kill(os.getpid(), 9)
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomAffine, ToTensor, ToPILImage
from utils.data import PixelCorruption, AugmentedDataset,SingleViewDataset
from utils.evaluation_new import evaluate, split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import os
from tensorboardX import SummaryWriter

mnist_dir = '.MNIST'
train_set = MNIST(mnist_dir, download=True, train=True, transform=ToTensor())
test_set = MNIST(mnist_dir, download=True, train=False, transform=ToTensor())
device_test = 'cpu' #'cuda'
experiment_dir = './new_experiments/MIB_new'
pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
             and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))
print(pretrained)
resume_training =False
if resume_training:
    load_model_file = os.path.join(experiment_dir, 'model.pt')
    config_file = os.path.join(experiment_dir, 'config.yml')
logging = True
if logging:
    writer = SummaryWriter(log_dir=experiment_dir)
else:
    os.makedirs(experiment_dir, exist_ok=True)
    writer = None
    
# Defining the augmentations
t = Compose([
    RandomAffine(degrees=15,
                 translate=[0.1, 0.1],
                 scale=[0.9, 1.1],
                 shear=15),  # Small affine transformations
    ToTensor(),              # Conversion to torch tensor
    PixelCorruption(0.8)     # PixelCorruption with keep probability 80%
])

# Creating the multi-view dataset using the augmentation class defined by t
mv_train_set = SingleViewDataset(MNIST(mnist_dir, download=True), t)

# Initialization of the data loader
train_loader = DataLoader(mv_train_set, batch_size=64, shuffle=True, num_workers=8)

import training as training_module
TrainerClass = getattr(training_module, 'MIB_newTrainer')
trainer = TrainerClass( writer=writer,lr=1e-6,beta_value=1e-6)
trainer.to(device_test)
train_subset = split(mv_train_set, 100, 'Balanced')
test_set = split(mv_train_set, 100, 'Random')
checkpoint_count = 0

from tqdm import tqdm
for epoch in tqdm(range(500)):
    for data in tqdm(train_loader):
        trainer.train_step(data)
    train_accuracy, test_accuracy = evaluate(encoder=trainer.encoder_v1, train_on=train_subset, test_on=test_set,
                                                 device=device_test)
    tqdm.write('Train Accuracy: %f' % train_accuracy)
    tqdm.write('Test Accuracy: %f' % test_accuracy)
    writer.add_scalar(tag='evaluation/train_accuracy', scalar_value=train_accuracy, global_step=trainer.iterations)
    writer.add_scalar(tag='evaluation/test_accuracy', scalar_value=test_accuracy, global_step=trainer.iterations)
    trainer._log_loss()
    if epoch%1 == 0 :
        #tqdm.write('Storing model checkpoint')
        while os.path.isfile(os.path.join(experiment_dir,'checkpoint_%d.pt'%checkpoint_count)):
            checkpoint_count+=1
        trainer.save(os.path.join(experiment_dir,'checkpoint_%d.pt'%checkpoint_count))
        checkpoint_count+=1