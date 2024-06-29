import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import FoundNANsError
from utils_train import get_model, make_dataset
from lib import round_columns
import lib

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:0'),
    seed = 0,
    change_val = False
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

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
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, 
        num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, 
        scheduler=scheduler, 
        device=device,
        distribution_types=distribution_types,
        distribution_params=distribution_params
    )

    diffusion.to(device)
    diffusion.eval()
    
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
            x_gen.append(x_temp)
            y_gen.append(y_temp)
        
        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)