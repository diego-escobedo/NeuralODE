import numpy as np
import matplotlib.pyplot as plt
import torch

def make_image(n=10000):
    """Make an X shape."""
    points = np.zeros((n,2))
    points[:n//2,0] = np.linspace(-1,1,n//2)
    points[:n//2,1] = np.linspace(1,-1,n//2)
    points[n//2:,0] = np.linspace(1,-1,n//2)
    points[n//2:,1] = np.linspace(1,-1,n//2)
    np.random.seed(42)
    noise = np.clip(np.random.normal(scale=0.1, size=points.shape),-0.2,0.2)
    np.random.seed(None)
    points += noise
    img, _ = np.histogramdd(points, bins=40, range=[[-1.5,1.5],[-1.5,1.5]])
    return img


class ImageDataset():
    """Sample from a distribution defined by an image."""

    def __init__(self, img, MAX_VAL=4.0, thresh=0):
        img[img<thresh]=0; # threshold to cut empty region of image
        
        h, w = img.shape
        xx = np.linspace(-MAX_VAL, MAX_VAL, w)
        yy = np.linspace(-MAX_VAL, MAX_VAL, h)
        xx, yy = np.meshgrid(xx, yy)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        self.means = np.concatenate([xx, yy], 1)
        self.probs = img.reshape(-1); 
        self.probs /= self.probs.sum();
        self.noise_std = np.array([MAX_VAL/w, MAX_VAL/h])

    def sample(self, batch_size=512, normalize = False):
        inds = np.random.choice(int(self.probs.shape[0]), int(batch_size), p=self.probs)
        m = self.means[inds]
        samples = np.random.randn(*m.shape) * self.noise_std + m
        return torch.from_numpy(samples).type(torch.FloatTensor)


def import_img(file):
    """
    file : str
        filename for an rgba image
    Returns
    gimg : 2D array
        greyscale image
    """
    img = plt.imread(file)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gimg = np.dot(img[...,:3], rgb_weights)
    return gimg

class BoundingBox():
    ## use like:
    # BB = BoundingBox(z_target);
    # smps = BB.sampleuniform(t_N = 30, x_N = 10, y_N = 11, z_N=12, bbscale = 1.1);
    # smps = BB.samplerandom(N = 10000, bbscale = 1.1);
    
    def __init__(self, z_target_full, device=None):
        self.T = z_target_full.shape[0]; 
        self.dim = z_target_full.shape[2]
        self.device = device
        
        # min corner, max corner, center
        self.mic = z_target_full.reshape(-1,self.dim).min(0)[0]
        self.mac = z_target_full.reshape(-1,self.dim).max(0)[0]; 
        self.C = (self.mic+self.mac)/2; 
        
    def extendedBB(self, bbscale):
        # extended bounding box.
        emic = (self.mic-self.C)*bbscale+self.C; 
        emac = (self.mac-self.C)*bbscale+self.C; 
        return emic, emac
        
    def sampleuniform(self, t_N = 30, x_N = 10, y_N = 11, z_N = 12, bbscale = 1.1):
        [eLL,eTR] = self.extendedBB(bbscale)
        
        tspace = torch.linspace(0, self.T-1, t_N)
        xspace = torch.linspace(eLL[0], eTR[0], x_N)
        yspace = torch.linspace(eLL[1], eTR[1], y_N)
        if self.dim == 3:
            zspace = torch.linspace(eLL[2], eTR[2], z_N)
            xgrid,ygrid,zgrid,tgrid=torch.meshgrid(xspace,yspace,zspace,tspace)
            z_sample = torch.transpose(torch.reshape(torch.stack([tgrid,xgrid,ygrid,zgrid]),(4,-1)),0,1).to(self.device)
        else:
            xgrid,ygrid,tgrid=torch.meshgrid(xspace,yspace,tspace)
            z_sample = torch.transpose(torch.reshape(torch.stack([tgrid,xgrid,ygrid]),(3,-1)),0,1).to(self.device)
        
        return z_sample.to(self.device)
    
    def samplerandom(self, N = 10000, bbscale = 1.1):
        [eLL,eTR] = self.extendedBB(bbscale)
        # time goes from 0 to T-1
        dT = torch.Tensor([self.T-1]).to(self.device); # size of time begin to end
        TC = torch.Tensor([(self.T-1.0)/2.0]).to(self.device); # time center
        
        z_sample = torch.rand(N, self.dim + 1).to(self.device)-0.5
        deltx = torch.cat((dT,eTR-eLL))
        z_sample = deltx*z_sample + torch.cat((TC,self.C))

        return z_sample