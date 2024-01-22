import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR  # Import StepLR scheduler
from CCRLDataset import CCRLDataset
from AlphaZeroNetwork import AlphaZeroNet
import matplotlib.pyplot as plt

# Training params
num_epochs = 30
num_blocks = 20
num_filters = 256
ccrl_dir = "/data/acp22np/ScalableML/Alpha-zero/selfplay12/"
logmode = True

# Check for CUDA availability and set the device accordingly
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

def train():
    train_ds = CCRLDataset(ccrl_dir)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)

    # Load your pretrained model here
    pretrained_model_path = "/data/acp22np/ScalableML/Alpha-zero/weights/human_20x256.pt"
    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters).to(device)
    alphaZeroNet.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu')))

    alphaZeroNet.train()

    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=0.001)
    mseLoss = nn.MSELoss()

    # Create the scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    print('Starting fine-tuning')

    value_losses_epoch = []
    policy_losses_epoch = []

    for epoch in range(num_epochs):
        
        value_losses = []
        policy_losses = []

        for iter_num, data in enumerate(train_loader):
            if data is None:
                continue

            optimizer.zero_grad()

            position = data['position'].to(device)
            valueTarget = data['value'].to(device)
            policyTarget = data['policy'].to(device)

            valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget,
                                                policyTarget=policyTarget)

            loss = valueLoss + policyLoss

            value_losses.append(float(valueLoss))
            policy_losses.append(float(policyLoss))

            loss.backward()

            optimizer.step()

            message = 'Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                epoch, iter_num, len(train_loader), float(valueLoss), float(policyLoss))

            if iter_num % 10 == 0:
                if iter_num != 0:
                    print(('\b' * len(message)), end='')
                print(message, end='', flush=True)
                if logmode:
                    print('')

        avg_value_loss = sum(value_losses) / len(value_losses)

        value_losses_epoch.append(avg_value_loss)
        policy_losses_epoch.append(sum(policy_losses) / len(policy_losses))

        print('')

        networkFileName = '/data/acp22np/ScalableML/Alpha-zero/models/FineTuned_human_{}x{}_epoch{}.pt'.format(num_blocks, num_filters, epoch)
        torch.save(alphaZeroNet.state_dict(), networkFileName)
        print('Saved fine-tuned model to {}'.format(networkFileName))
        
    # Plotting
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(value_losses_epoch, label="Value Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Value Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(policy_losses_epoch, label="Policy Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Policy Loss over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("/data/acp22np/ScalableML/Output/self_epochVSloss.png")

if __name__ == '__main__':
    train()
