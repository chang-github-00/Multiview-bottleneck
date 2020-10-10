import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from spectral_normalization import SpectralNorm


# Encoder architecture           # what's parent class
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(28 * 28 , 1024),  #notice 新模型这里28*28*2
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, z_dim * 2),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        params = self.net(x)       # input x to vanilla MLP

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7  # Make sigma always positive

        return Independent(Normal(loc=mu, scale=sigma), 1)  # Return a factorized Normal distribution


class Decoder(nn.Module):
    def __init__(self, z_dim, scale=0.39894):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.scale = scale

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28)
        )

    def forward(self, z):     #what's this?
        x = self.net(z)
        return Independent(Normal(loc=x, scale=self.scale), 1)


# Auxiliary network for mutual information estimation
class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1
    
    
class ConditionMIEstimator(nn.Module):
    def __init__(self,size1,size2,size3): # I(X1;X2|X3)
        super(ConditionMIEstimator,self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(size1+size2+size3,1024),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),  
        )
        
    def forward(self,x1,x2,x3):    # 可能有问题 ！！！
        pos = self.net(torch.cat([x1,x2.view(x2.shape[0], -1),x3.view(x3.shape[0], -1)],1))  # concatenate x1,x2,x3
        neg = self.net(torch.cat([torch.roll(x1,1,0),x2.view(x2.shape[0], -1),x3.view(x3.shape[0], -1)],1)) # concatenate x1.rand ,x2,x3
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean()-neg.exp().mean()+1


class WDEstimator(nn.Module): #Wasserstein dependency estimator
    def __init__(self, size1, size2):
        super(WDEstimator, self).__init__()

        # Vanilla MLP
        self.net = nn.Sequential(
            SpectralNorm(nn.Linear(size1 + size2, 1024)),
            SpectralNorm(nn.ReLU(True)),
            SpectralNorm(nn.Linear(1024, 1024)),
            SpectralNorm(nn.ReLU(True)),
            SpectralNorm(nn.Linear(1024, 1)),
        )

    # Gradient for JSD mutual information estimation and EB-based estimation
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))  # Positive Samples
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1
