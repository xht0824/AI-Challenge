import os 
import sys 
import torch
import torch.nn as nn
from torchvision import transforms, datasets 
import torch.optim as optim 
from tqdm import tqdm  
from model import MLP,FNORecon
from dataset import Mydataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # please use tensorboard to see the change of train loss
    writer = SummaryWriter(log_dir='runs/FNO/luna')
    print("using {} device.".format(device))
    # dataloader
    # you can choose "data_crm_example.npy" / "data_Sphere_stationary.npy" / "data_agard.npy" / "data_luna.npy"
    data = Mydataset('data_luna.npy')
    train_size = int (0.4 * len(data))
    val_size = int (0.2 * len(data))
    test_size = len(data) - train_size - val_size
    batch_size = 16
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)
    print("using {} points for training, {} points for validation.".format(train_size, val_size))

    # model setting
    net = FNORecon(sensor_num=5).to(device)
    #net = MLP().to(device)
    loss_function = nn.L1Loss()  # MAELoss
    lr = 0.001
    optimizer = optim.Adam(net.parameters(), lr=lr)
    epochs = 10
    save_path = os.path.abspath(os.path.join(os.getcwd(), './results/weights/fno'))
    if not os.path.exists(save_path):    
        os.makedirs(save_path)
    sample_num = 0
    for epoch in range(epochs):
        ############################################################## train ######################################################
        net.train()            
        train_loss = 0
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for data in train_bar :
            features, labels = data 
            features = features.reshape(features.shape[0], 1, 5)
            sample_num += features.shape[0]
            output = net(features.to(device)).squeeze(-1).squeeze(-1)
            loss = loss_function(output, labels.to(device))
            train_loss += loss.sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar(tag="train_loss",
                      scalar_value=loss,  
                      global_step=sample_num  
                      )
        net.eval()
        val_loss = 0
        with torch.no_grad(): 
            for val_data in val_loader:
                features, labels = val_data
                features = features.reshape(features.shape[0], 1, 5)
                output = net(features.to(device)).squeeze(-1).squeeze(-1)
                loss = loss_function(output, labels.to(device))
                val_loss += loss.sum().item()
        
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f' %  (epoch + 1, train_loss / train_size * batch_size, val_loss / val_size * batch_size))   
    torch.save(net.state_dict(), save_path + '/fno_luna.pt')
    test_loss = 0
    for data in test_loader:
        features, labels = data
        features = features.reshape(features.shape[0], 1, 5)
        output = net(features.to(device)).squeeze(-1).squeeze(-1)
        loss = loss_function(output, labels.to(device))
        test_loss += loss.sum().item()
    
    print('test_loss: %.3f' % (test_loss / test_size * batch_size))
     

if __name__ == '__main__':
    main()

