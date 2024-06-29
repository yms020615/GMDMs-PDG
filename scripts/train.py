import sys
sys.path.append('/content/drive/MyDrive/TabDDPM')

from copy import deepcopy
import torch
import os
import numpy as np
import zero
from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset, update_ema
import lib
import pandas as pd

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:0')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 100
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1

def train(
    parent_dir,
    real_data_path = 'data/higgs-small',
    steps = 1000,
    lr = 0.002,
    weight_decay = 1e-4,
    batch_size = 1024,
    model_type = 'mlp',
    model_params = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False,
    distribution_types=None,  # 추가
    distribution_params=None  # 추가
):
    real_data_path = os.path.normpath(real_data_path)
    parent_dir = os.path.normpath(parent_dir)

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)

    dataset = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])
    print(K)

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    model_params['d_in'] = d_in
    print(d_in)
    
    print(model_params)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )
    model.to(device)

    train_loader = lib.prepare_fast_dataloader(dataset, split='train', batch_size=batch_size)

    '''
    distribution_types = [
        'gaussian',  # 0(x)
        'gmm1',  # 1(x)
        'gaussian',  # 2(x)
        'gaussian',  # 3(x)
        'gaussian',  # 4(x)
        'gaussian',  # 5(x)
        'gmm6',  # 6(x)
        'gmm7',  # 7(x)
        'gaussian'  # 8(x)
    ]

    distribution_params = [
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 0(x) - Gamma
        {'gmm1': [
            (0.074, -0.2, 0.95),
            (0.244, -0.1, 1.05),
            (0.283, 0.0, 1.1),
            (0.283, 0.1, 1.05),
            (0.116, 0.2, 0.95)
        ]},  # 1(x) - GMM
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 2(x) - Gamma
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 3(x) - Gamma
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 4(x) - Gamma
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 5(x) - GMM
        {'gmm6': [
            (0.062, -0.2, 0.95),
            (0.455, -0.1, 1.15),
            (0.158, 0.0, 1.2),
            (0.248, 0.1, 1.15),
            (0.077, 0.2, 0.95)
        ]},  # 6(x) - GMM
        {'gmm7': [
            (0.245, -0.2, 1.1),
            (0.142, -0.1, 1.05),
            (0.107, 0.0, 1.1),
            (0.405, 0.1, 1.05),
            (0.101, 0.2, 1.1)
        ]},  # 7(x) - GMM
        {'gaussian': [(1.0, 0.0, 1.0)]}  # 8(x) - Gamma
    ]
    '''
        
    distribution_types = [
        'gaussian',       # 0(x) - GMM
        'gaussian',       # 1(x) - GMM
        'gaussian', # 2(x) - Multinomial
        'gaussian',       # 3(x) - GMM
        'gaussian', # 4(x) - Multinomial
        'gaussian',      # 5(x) - GMM (approximation of Uniform)
        'gaussian',      # 6(x) - GMM (approximation of Uniform)
    ]

    distribution_params = [
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 0(x) - Gaussian
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 1(x) - Gaussian
        {'gmm2': [
            (0.0392, -0.2, 0.9),
            (0.1020, -0.1, 1.0),
            (0.1098, 0.0, 1.05),
            (0.1011, 0.1, 1.0),
            (0.1006, 0.2, 0.9),
            (0.1002, -0.2, 0.9),
            (0.0970, -0.1, 1.0),
            (0.1038, 0.0, 1.05),
            (0.0989, 0.1, 1.0),
            (0.0980, 0.2, 0.9),
            (0.0494, -0.2, 0.9)
        ]},  # 2(x) - GMM
        {'gaussian': [(1.0, 0.0, 1.0)]},  # 3(x) - Gaussian
        {'gmm4': [
            (0.5089, -0.04, 0.95),
            (0.4570, -0.01, 1.0),
            (0.0281, 0.0, 1.05),
            (0.0060, 0.01, 1.0)
        ]},  # 4(x) - GMM
        {'gmm5': [
            (6/41, -0.15, 0.95),
            (5/41, -0.12, 1.0),
            (4/41, -0.09, 1.05),
            (3/41, -0.06, 1.0),
            (2/41, -0.03, 0.95),
            (1/41, 0.0, 1.0),
            (2/41, 0.03, 1.05),
            (3/41, 0.06, 1.0),
            (4/41, 0.09, 0.95),
            (5/41, 0.12, 1.0),
            (6/41, 0.15, 1.05)
        ]},  # 5(x) - GMM (approximation of Uniform)
        {'gmm6': [
            (6/41, -0.15, 0.95),
            (5/41, -0.12, 1.0),
            (4/41, -0.09, 1.05),
            (3/41, -0.06, 1.0),
            (2/41, -0.03, 0.95),
            (1/41, 0.0, 1.0),
            (2/41, 0.03, 1.05),
            (3/41, 0.06, 1.0),
            (4/41, 0.09, 0.95),
            (5/41, 0.12, 1.0),
            (6/41, 0.15, 1.05)
        ]}  # 6(x) - GMM (approximation of Uniform)
    ]

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=gaussian_loss_type,
        num_timesteps=num_timesteps,
        scheduler=scheduler,
        device=device,
        distribution_types=distribution_types,
        distribution_params=distribution_params
    )
    diffusion.to(device)
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=lr,
        weight_decay=weight_decay,
        steps=steps,
        device=device
    )
    trainer.run_loop()

    trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
