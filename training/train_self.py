"""
Fine-tune a pretrained AlphaZeroNet on self-play games (the Hybrid approach).

Example:
    python training/train_self.py --data-dir selfplay_games/ \
        --base-model weights/human_20x256.pt --output-dir checkpoints/
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CCRLDataset import CCRLDataset
from AlphaZeroNetwork import AlphaZeroNet

# Model architecture
num_blocks = 20
num_filters = 256

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


def train(args):
    train_ds = CCRLDataset(args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    alphaZeroNet = AlphaZeroNet(num_blocks, num_filters).to(device)
    alphaZeroNet.load_state_dict(torch.load(args.base_model, map_location=torch.device('cpu')))

    alphaZeroNet.train()

    optimizer = optim.Adam(alphaZeroNet.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(args.output_dir, exist_ok=True)

    print('Starting fine-tuning')

    value_losses_epoch = []
    policy_losses_epoch = []

    for epoch in range(args.epochs):

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

            if iter_num % 10 == 0:
                print('Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                    epoch, iter_num, len(train_loader), float(valueLoss), float(policyLoss)))

        scheduler.step()

        value_losses_epoch.append(sum(value_losses) / len(value_losses))
        policy_losses_epoch.append(sum(policy_losses) / len(policy_losses))

        networkFileName = os.path.join(
            args.output_dir,
            'FineTuned_{}x{}_epoch{}.pt'.format(num_blocks, num_filters, epoch))
        torch.save(alphaZeroNet.state_dict(), networkFileName)
        print('Saved fine-tuned model to {}'.format(networkFileName))

    # Plot the loss curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
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
        plt.savefig(os.path.join(args.output_dir, 'epoch_vs_loss.png'))
    except ImportError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune AlphaZeroNet on self-play PGN data.')
    parser.add_argument('--data-dir', required=True, help='Directory of self-play PGN games')
    parser.add_argument('--base-model', required=True, help='Pretrained .pt weights to start from')
    parser.add_argument('--output-dir', default='checkpoints', help='Where to save checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    train(parser.parse_args())
