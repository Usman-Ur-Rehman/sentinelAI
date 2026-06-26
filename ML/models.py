from torch import nn

class AutoEncoder(nn.Module):

    def __init__(self,Inputdim):
        super().__init__()
        self.encoder=nn.sequential(
            nn.Linear(Inputdim,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,8)
        )

        self.decoder=nn.sequential(
            nn.Linear(8,16),    
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,Inputdim)
        )

    def runner(self,x):
        return self.decoder(self.encoder(x))
