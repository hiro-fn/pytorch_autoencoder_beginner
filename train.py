import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image

from net import AENet

image_size = 28

def create_transform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def set_dataset(dataset_path, transform, options):
    batch_size = options['batch_size']
    num_worker = options['num_workers']
    my_datasets = datasets.ImageFolder(root=dataset_path, transform=transform)
    # my_datasets = datasets.MNIST(root='/data', transform=transform)

    print(f'dataset : {len(my_datasets)}')

    return torch.utils.data.DataLoader(my_datasets, batch_size=batch_size,
                                        shuffle=True, num_workers=num_worker)

def to_image(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, image_size, image_size)
    return x

def run_train(model, dataset, options):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=options['lr'],
                          weight_decay=options['decay'])

    for epoch in range(options['epoch']):
        running_loss = 0.0
        print(epoch)

        for data in dataset.dataset:
            raw_inputs, _ = data
            convert_image = raw_inputs.view(raw_inputs.size(0), -1)
            inputs = convert_image.to('cuda')

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

        if (epoch % 1 == 0):
            convert_image = to_image(outputs.to('cpu').data)
            save_image(convert_image, f'image\\{epoch}.png')

        torch.save(model, f'model\\{epoch}')

def main():
    train_dataset_path = "D:\\project\\idol_classification\\images\\train\\"

    options = {
        'batch_size': 4,
        'epoch': 300,
        'lr': 1e-8,
        'num_workers': 4,
        'decay': 1e-5
    }

    data_transform = create_transform()
    train_dataset = set_dataset(train_dataset_path, data_transform, options)

    model = AENet(image_size).to('cuda')
    run_train(model, train_dataset, options)

if __name__ == '__main__':
    main()