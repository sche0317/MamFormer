from torchmetrics.image import StructuralSimilarityIndexMeasure
from torcheval.metrics.functional import peak_signal_noise_ratio as psnr_1
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from model.Mamformer import mamba_vision_T
import matplotlib.pyplot as plt
from scipy.stats import bootstrap

model = mamba_vision_T()
model.load_state_dict(torch.load('./checkpoint/t2_pd/mf_t_31.841.pth'))
model.cuda()
ssim = StructuralSimilarityIndexMeasure().cuda()

x_valid_set = torch.load('./data/brats_data/flair_gaussian_test.pth')
y_valid_set = torch.load('./data/brats_data/t2_gaussian_test.pth')
test_dataset = TensorDataset(x_valid_set, y_valid_set)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)


def compute_metrics(generated_volume, target_volume):

    average_ssim = ssim(generated_volume, target_volume)
    average_psnr = psnr_1(generated_volume, target_volume)
    return average_psnr, average_ssim

def main():
    test_psnr_results = []
    test_ssim_results = []
    model.eval()
    with torch.no_grad():
        for x_valid, y_valid in test_loader:
            x_valid = x_valid.cuda()
            y_valid = y_valid.cuda()
            pred_valid = model(x_valid)
            psnr_value, ssim_value = compute_metrics(pred_valid, y_valid)
            test_psnr_results.append(psnr_value)
            test_ssim_results.append(ssim_value)

    test_avg_psnr = sum(test_psnr_results) / len(test_psnr_results)
    test_avg_ssim = sum(test_ssim_results) / len(test_ssim_results)

    psnr_metric = np.array([t.to('cpu').detach().numpy() for t in test_psnr_results])
    ssim_metric = np.array([t.to('cpu').detach().numpy() for t in test_ssim_results])
    print(f'Test PSNR: {test_avg_psnr:.4f}, Test SSIM: {test_avg_ssim:.4f}')
    return psnr_metric, ssim_metric

def psnr_bootstrap(psnr_metric):
    rng = np.random.default_rng()
    data = (psnr_metric,)  # samples must be in a sequence
    res_1 = bootstrap(data, np.mean, random_state=rng)
    fig, ax = plt.subplots()
    ax.hist(res_1.bootstrap_distribution, bins=30, color='green')
    ax.set_title('Bootstrap Distribution of PSNR')
    ax.set_xlabel('Mean value')
    ax.set_ylabel('Frequency')
    plt.savefig(
        'f_t2_psnr.png',
        format='png',
        dpi=600,
        bbox_inches='tight',
        transparent=False
    )
    plt.show()

def ssim_bootstrap(ssim_metric):
    rng = np.random.default_rng()
    data = (ssim_metric,)  # samples must be in a sequence
    res_1 = bootstrap(data, np.mean, random_state=rng)
    fig, ax = plt.subplots()
    ax.hist(res_1.bootstrap_distribution, bins=30, color='green')
    ax.set_title('Bootstrap Distribution of SSIM')
    ax.set_xlabel('Mean value')
    ax.set_ylabel('Frequency')
    plt.savefig(
        'f_t2_ssim.png',
        format='png',
        dpi=600,
        bbox_inches='tight',
        transparent=False
    )
    plt.show()



if __name__ == "__main__":

    psnr_eval, ssim_eval =  main()
    psnr_bootstrap(psnr_eval)
    ssim_bootstrap(ssim_eval)