import sys, math, random, copy
from typing import Union, Callable, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from einops import rearrange
from models.encoder import CoSTEncoder
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan

class PretrainDataset(Dataset):

    def __init__(self,
                 data,
                 sigma,
                 p=0.5,
                 multiplier=10):
        super().__init__()
        self.data = data
        self.p = p
        self.sigma = sigma
        self.multiplier = multiplier
        self.N, self.T, self.D = data.shape

    def __getitem__(self, item):
        ts = self.data[item % self.N]
        return self.transform(ts), self.transform(ts)

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        return self.scale(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma)


class MFF_FTModel(nn.Module):
    def __init__(self,
                 encoder_q: nn.Module, encoder_k: nn.Module,
                 kernels: List[int],
                 device: Optional[str] = 'cuda',
                 dim: Optional[int] = 128,
                 alpha: Optional[float] = 0.05,
                 K: Optional[int] = 65536,
                 m: Optional[float] = 0.999,
                 T: Optional[float] = 0.07):
        super().__init__()

        self.K = K
        self.m = m
        self.T = T
        self.device = device

        self.kernels = kernels

        self.alpha = alpha

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.register_buffer('queue', F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.liner = nn.AdaptiveAvgPool1d(2*dim)

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def time_frequency_contrastive_loss(self, T_z, F_z, t):
        positive_sc = torch.sum(torch.mul(T_z, F_z), dim=-1)
        positive_sc = F.normalize(positive_sc, dim=1)
        positive_sc = torch.exp(positive_sc)
        negative_sc = torch.matmul(T_z, F_z.transpose(1, 2))
        negative_sc = F.normalize(negative_sc, dim=-1)
        negative_sc = torch.exp(negative_sc)
        negative_sc = torch.sum(negative_sc, dim=-1)
        loss = - torch.sum((torch.log(positive_sc / negative_sc)))
        return(self.alpha * loss)

    def forward(self, x_q, x_k):
        rand_idx = np.random.randint(0, x_q.shape[1])
        q_t, q_s, x_feature_extractor = self.encoder_q(x_q)
        frequency_trend_z = torch.cat([q_s, q_t], dim=-1)
        frequency_trend_z = self.liner(frequency_trend_z)
        loss = self.time_frequency_contrastive_loss(x_feature_extractor, frequency_trend_z, self.T)

        q_s = F.normalize(q_s, dim=-1)
        _, k_s, _ = self.encoder_q(x_k)
        k_s = F.normalize(k_s, dim=-1)
        q_s_freq = fft.rfft(q_s, dim=1)
        k_s_freq = fft.rfft(k_s, dim=1)
        q_s_amp, q_s_phase = self.convert_coeff(q_s_freq)
        k_s_amp, k_s_phase = self.convert_coeff(k_s_freq)

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase,k_s_phase)
        loss += (self.alpha * (seasonal_loss/2))

        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr


class MFF_FTNet:
    def __init__(self,
                 input_dims: int,
                 kernels: List[int],
                 alpha: bool,
                 max_train_length: int,
                 output_dims: int = 320,
                 hidden_dims: int = 64,
                 depth: int = 10,
                 device: 'str' ='cuda',
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 dim: int = 64,
                 batch_size: int = 16,
                 after_iter_callback: Union[Callable, None] = None,
                 after_epoch_callback: Union[Callable, None] = None):

        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.device = device
        self.depth = depth
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.dim = dim

        if kernels is None:
            kernels = []

        self.net = CoSTEncoder(
            input_dims=input_dims, output_dims=output_dims,
            kernels=kernels,
            dim=self.dim,
            length=max_train_length,
            hidden_dims=self.hidden_dims, depth=self.depth,
        ).to(self.device)

        self.mff = MFF_FTModel(
            self.net,
            copy.deepcopy(self.net),
            kernels=kernels,
            dim=self.net.component_dims,
            alpha=alpha,
            K=256,
            device=self.device,
        ).to(self.device)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0

    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600

        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        multiplier = 1 if train_data.shape[0] >= self.batch_size else math.ceil(self.batch_size / train_data.shape[0])
        train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), sigma=0.5, multiplier=multiplier)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        optimizer = torch.optim.SGD([p for p in self.mff.parameters() if p.requires_grad],
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                x_q, x_k = map(lambda x: x.to(self.device), batch)
                if self.max_train_length is not None and x_q.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x_q.size(1) - self.max_train_length + 1)
                    x_q = x_q[:, window_offset : window_offset + self.max_train_length]
                    x_k = x_k[:, window_offset : window_offset + self.max_train_length]
                optimizer.zero_grad()
                loss = self.mff(x_q, x_k)
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

                if n_iters is not None:
                    adjust_learning_rate(optimizer, self.lr, self.n_iters, n_iters)
            if interrupted:
                break
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            # if verbose:
            #     # print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            #     print(f"Epoch #{self.n_epochs}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)

            if n_epochs is not None:
                adjust_learning_rate(optimizer, self.lr, self.n_epochs, n_epochs)
        self.n_epochs = 0
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out_t, out_s, _ = self.net(x.to(self.device, non_blocking=True))
        out = torch.cat([out_t[:, -1], out_s[:, -1]], dim=-1)
        return rearrange(out.cpu(), 'b d -> b () d')
    
    def encode(self, data, mode, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        if mode == 'forecasting':
            encoding_window = None
            slicing = None
        else:
            raise NotImplementedError(f"mode {mode} has not been implemented")

        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slicing,
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slicing,
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)

        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)


def adjust_learning_rate(optimizer, lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
