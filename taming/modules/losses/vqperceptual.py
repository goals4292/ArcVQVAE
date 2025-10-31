import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


# ArcLoss
class ArcLoss(nn.Module):
    def __init__(self, s=10.0, m=0.1, top_k=8):
        super().__init__()
        self.s = s
        self.m = m
        self.top_k = top_k

    def forward(self, z_e, codebook_weight):
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW, C)
        z_e_flat = F.normalize(z_e_flat, p=2, dim=1)

        e_norm = F.normalize(codebook_weight.detach(), p=2, dim=1)  # (K, C)

        cos_theta = torch.matmul(e_norm, z_e_flat.t())  # (K, BHW)

        topk_val, topk_idx = torch.topk(cos_theta, self.top_k, dim=1)

        theta = torch.acos(torch.clamp(topk_val, -1.0 + 1e-7, 1.0 - 1e-7))
        cos_theta_m = torch.cos(theta + self.m)  # (K, top_k)

        # Clone logits
        logits = cos_theta.clone()

        # Replace the top-K values with margin-applied values
        row_idx = torch.arange(e_norm.size(0)).unsqueeze(1).expand(-1, self.top_k)  # (K, top_k)
        logits[row_idx, topk_idx] = cos_theta_m

        # Scale
        logits = self.s * logits  # (K, BHW)

        numerator = torch.sum(torch.exp(logits[row_idx, topk_idx]), dim=1)  # (K,)
        denominator = torch.sum(torch.exp(logits), dim=1)  # (K,)

        loss = -torch.log(numerator / denominator).mean()

        return loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", arc_gamma=2.0, arc_gamma_decay=1e-4):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        

        # ArcLoss 도입
        self.arc_loss = ArcLoss(s=10.0, m=0.1, top_k=8)
        self.gamma = arc_gamma
        self.gamma_decay = arc_gamma_decay

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                z_e_x=None, codebook_weight=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            # ArcLoss 추가
            if z_e_x is not None and codebook_weight is not None:
                gamma_t = self.gamma * torch.exp(-self.gamma_decay * torch.tensor(global_step, dtype=torch.float32, device=inputs.device))
                arc_loss = self.arc_loss(z_e_x, codebook_weight)
                loss += gamma_t * arc_loss
            else:
                print("error")
                arc_loss = torch.tensor(0.0).to(inputs.device)
                gamma_t = torch.tensor(0.0).to(inputs.device)

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/arc_loss".format(split): arc_loss.detach().mean(),
                   "{}/gamma".format(split): gamma_t.detach(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
