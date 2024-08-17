import torch
import math
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import time
from tqdm import tqdm
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr
from functools import partial
from ml_collections import config_dict
from model.Mamformer import mamba_vision_T

x_set = torch.load('./data/t1_f_norm.pt')
y_set = torch.load('./data/pd_f_norm.pt')
dataset = TensorDataset(x_set, y_set)
train_dataset, valid_dataset = random_split(
    dataset=dataset,
    lengths=[20000, 3000],
    generator=torch.Generator().manual_seed(0)
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True, drop_last=True)

model = mamba_vision_T()
model.cuda()

def get_lr_basic(it, config):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def compute_metrics(generated_volume, target_volume):

    average_psnr = psnr(generated_volume, target_volume)
    return average_psnr


def train(train_loader, valid_loader, model, optimizer, loss_function, config, iter_num, get_lr):
    model.train()
    st = time.time()
    opt = optimizer[0]
    loss_fn = loss_function
    for i, (x, y) in enumerate(train_loader):
        lr = get_lr(iter_num)
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        x = x.cuda()
        y = y.cuda()
        pred = model(x)
        loss = 2 * loss_fn(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # valid
        iter_num += 1
        if iter_num % 125 == 0:

            results = []
            model.eval()
            with torch.no_grad():
                for x_valid, y_valid in valid_loader:
                    x_valid = x_valid.cuda()
                    y_valid = y_valid.cuda()
                    pred_valid = model(x_valid)
                    psnr = compute_metrics(pred_valid, y_valid)
                    results.append(psnr)

                avg_psnr = sum(results) / len(results)

            model.train()
            print(f"step {iter_num}, "
                  f"lr: {lr:.7f}, "
                  f"loss: {loss:.4f}, "
                  f"avg_psnr: {avg_psnr:.4f}, "
                  f"consume {time.time() - st:.2f}s")

            st = time.time()
            if iter_num >= config.max_iters:
                break
    return iter_num


def train_config():
    config = config_dict.ConfigDict()
    config.learning_rate = 5e-4
    config.min_lr = 1e-5
    config.warmup_iters = 3125
    config.max_iters = 31250
    config.lr_decay_iters = 31250
    config.num_epoch = 50
    config.batch_size = 32
    return config

if __name__ == "__main__":

    trained_epoch = 0
    config = train_config()
    print(config)
    print()

    get_lr = partial(get_lr_basic, config=config)
    loss_fn = torch.nn.L1Loss()
    optimizer = [torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999)), ]

    start_time = time.time()
    iter_num = trained_epoch * len(train_dataset) // (config.batch_size)
    for epoch in tqdm(range(trained_epoch, config.num_epoch)):
        if iter_num >= config.max_iters:
            break
        print(iter_num)
        iter_num = train(train_loader, valid_loader, model, optimizer, loss_fn, config, iter_num, get_lr)
        print('=' * 100)
        print(f'epoch: {epoch}, consume: {time.time() - start_time:.3f}s')
        print('=' * 60)

    end_time = time.time()
    run_time = end_time - start_time
    hour = int(run_time // 3600)
    minute = int((run_time - 3600 * hour) // 60)
    second = int(run_time - 3600 * hour - 60 * minute)

    print(f" Running time is ：{hour}hours{minute}minutes{second}seconds")