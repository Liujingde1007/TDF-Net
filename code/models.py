"""Time-domain fusion of a delay-domain denoiser for OFDM channel estimation.

Architecture (proposed, "TDF-Net" = Temporal Delay-Fusion Network)
-----------------------------------------------------------------
Input  : the pilot LS observation ``y/x`` on the pilot-bearing OFDM symbols of ``M``
         consecutive slots, de-rotated by the known pilot sequence and scaled by the known
         noise standard deviation.  ``M`` is the temporal window.

 1. **Delay-domain sparsity branch.**  Pilots are placed on a frequency comb, so the LS
    observation on each pilot symbol is a *decimated* sampling of the channel transfer
    function.  Since the channel has at most ``L_cp`` significant delay taps, the
    length-``N_f`` IDFT of the LS observation is concentrated in a few taps.  The branch
    computes this IDFT, keeps the first ``L_d`` taps (a hard sparsity prior implemented as a
    fixed, parameter-free linear operator) and embeds them into a feature vector.  This is
    the model-driven part of the network: it injects the physical structure instead of
    asking a generic CNN to relearn it from data.
 2. **Spatial encoder.**  A small fully-convolutional encoder with GroupNorm/LeakyReLU
    extracts multiscale features from the ``(N_f, N_t)`` pilot grid.
 3. **Temporal fusion.**  A GRU aggregates the pooled latent features of the ``M`` slots.
    Wireless channels are temporally correlated over a slot (Jakes correlation at
    100 km/h and 3.5 GHz is 0.18 at one slot and 0.43 at two slots), and the per-slot pilot
    overhead is too small to estimate the channel reliably from a single slot.  The GRU turns
    this redundancy into an explicit gain, and its state dimension is small (128) so the
    added cost is negligible and independent of ``M``.
 4. **Decoder** fuses the delay embedding with the temporal context at every resolution and
    reconstructs the full slot channel estimate.

Everything downstream of the fixed IDFT is trained end-to-end with an NMSE loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SYS as _SYS

# pilot layout of the configured system, used by the output-domain constraint
SYS_PILOT_SYMBOLS = tuple(_SYS.pilot_symbols)
SYS_PILOT_SPACING = _SYS.pilot_spacing


# ---------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------
def conv_block(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, padding_mode="replicate", bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.LeakyReLU(0.1, inplace=True),
    )


class DelayBranch(nn.Module):
    """Fixed-IDFT sparsity front end followed by a learnable embedding.

    The IDFT matrix is a buffer (no parameters).  A per-symbol learnable window is applied on
    top of the hard truncation so that the effective delay support can adapt to the channel
    during training while remaining interpretable.
    """

    def __init__(self, n_fft: int, n_pilot_sym: int, n_taps: int, emb_dim: int):
        super().__init__()
        self.n_fft = n_fft
        self.n_pilot_sym = n_pilot_sym
        self.n_taps = n_taps
        # IDFT along the frequency axis (normalised)
        k = torch.arange(n_fft, dtype=torch.float32)
        d = torch.arange(n_fft, dtype=torch.float32)
        W = torch.exp(2j * torch.pi * torch.outer(d, k) / n_fft) / n_fft  # (D, K)
        self.register_buffer("idft_re", W.real.contiguous())
        self.register_buffer("idft_im", W.imag.contiguous())
        self.window = nn.Parameter(torch.ones(1, 1, n_taps, 1))
        self.embed = nn.Sequential(
            nn.Conv2d(2, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(start_dim=2),                      # (B, 32, n_taps * N_t)
        )
        self.proj = nn.Sequential(
            nn.Linear(32 * n_taps, emb_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, N_f, N_t) -> delay embedding (B, emb_dim, 1, 1).

        Input planes: 0,1 = Re/Im of the LS observation scaled by 1/sigma; 2 = pilot mask;
        3 = normalised SNR.
        """
        b, _, nf, nt = x.shape
        zr = x[:, 0] * x[:, 2]                               # (B, N_f, N_t), zero off pilots
        zi = x[:, 1] * x[:, 2]

        # IDFT along frequency for each OFDM symbol
        a = zr.permute(0, 2, 1).reshape(-1, nf)              # (B*N_t, N_f)
        bq = zi.permute(0, 2, 1).reshape(-1, nf)
        wr = a @ self.idft_re.T - bq @ self.idft_im.T
        wi = a @ self.idft_im.T + bq @ self.idft_re.T
        # hard sparsity prior: retain only the first L_d delay taps (the physical support
        # of the channel is limited by the cyclic prefix)
        wr = wr[:, : self.n_taps]
        wi = wi[:, : self.n_taps]

        # (B*Nt, Ld) -> (B, 2, Ld, Nt): the two quadratures are the "channels"
        tap = torch.stack([wr, wi], dim=1).reshape(b, nt, 2, self.n_taps)
        tap = tap.permute(0, 2, 3, 1) * self.window
        h = self.embed(tap)                                  # (B, 32, Ld*Nt)
        h = h.view(b, 32, self.n_taps, nt).mean(dim=3)       # (B, 32, Ld)
        return self.proj(h.flatten(1)).view(b, -1, 1, 1)


class Encoder(nn.Module):
    def __init__(self, cin: int, base: int = 32, n_stages: int = 3):
        super().__init__()
        chs = [base * (2 ** i) for i in range(n_stages)]
        self.stem = conv_block(cin, chs[0])
        self.blocks = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = chs[0]
        for c in chs[1:]:
            self.down.append(nn.Conv2d(prev, c, 3, stride=2, padding=1, bias=False))
            self.blocks.append(conv_block(c, c))
            prev = c
        self.out_ch = prev

    def forward(self, x):
        feats = []
        h = self.stem(x)
        feats.append(h)
        for dn, blk in zip(self.down, self.blocks):
            h = blk(dn(h))
            feats.append(h)
        return feats


class Decoder(nn.Module):
    def __init__(self, enc_chs: list[int], n_skip: int, base: int = 32):
        super().__init__()
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for i in range(len(enc_chs) - 1, 0, -1):
            self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.fuse.append(conv_block(enc_chs[i] + enc_chs[i - 1] + n_skip, enc_chs[i - 1]))
        self.head = nn.Conv2d(enc_chs[0], 2, 3, padding=1)

    def forward(self, feats, ctx):
        h = feats[-1]
        for idx, (up, fuse) in enumerate(zip(self.up, self.fuse)):
            skip = feats[-2 - idx]
            h = up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            c = ctx.expand(-1, -1, skip.shape[-2], skip.shape[-1])
            h = fuse(torch.cat([h, skip, c], dim=1))
        return self.head(h)


class TDFNet(nn.Module):
    """Temporal Delay-Fusion Network.

    Parameters
    ----------
    n_fft : number of subcarriers
    n_sym : OFDM symbols per slot
    n_pilot_sym : number of pilot-bearing OFDM symbols
    n_taps : delay taps retained by the sparsity branch
    base_ch : base channel width
    seq_len : temporal window M (1 disables the temporal branch)
    constraint : how the delay-domain prior enters the estimate

        ``"feature"``  (default) the branch contributes a context vector only, so nothing
                       forces the output to lie in a delay subspace;
        ``"output"``   the branch *parameterises the estimate*: the network predicts K delay
                       coefficients, the channel is reconstructed from them as
                       :math:`\\mathbf{F}_K\\mathbf{g}`, and the decoder output is added as a
                       residual correction.  The residual head is zero-initialised, so the model
                       starts as a pure delay-domain least-squares estimator and can only
                       improve on it.  This is the model-driven formulation the classical
                       DFT-denoise estimator implicitly realises.
    """

    def __init__(self, n_fft: int = 72, n_sym: int = 14, n_pilot_sym: int = 2,
                 n_taps: int = 16, base_ch: int = 32, seq_len: int = 4,
                 hidden: int = 128, snr_emb: int = 32, use_delay: bool = True,
                 rnn: str = "gru", in_ch: int = 4, constraint: str = "feature"):
        super().__init__()
        self.n_fft, self.n_sym, self.n_pilot_sym, self.seq_len = n_fft, n_sym, n_pilot_sym, seq_len
        self.use_delay = use_delay
        self.in_ch = in_ch
        self.constraint = constraint if use_delay else "feature"
        # input planes: Re(LS/sigma), Im(LS/sigma), pilot mask, normalised SNR
        if use_delay:
            self.delay = DelayBranch(n_fft, n_pilot_sym, n_taps, hidden)
        else:
            self.delay = None
        self.encoder = Encoder(in_ch, base_ch, n_stages=3)
        enc_chs = [base_ch * (2 ** i) for i in range(3)]
        self.use_temporal = seq_len > 1
        if self.use_temporal:
            self.pre_gru = nn.Sequential(nn.Linear(enc_chs[-1], hidden), nn.LeakyReLU(0.1))
            cls = nn.GRU if rnn == "gru" else nn.LSTM
            self.rnn_name = rnn
            self.gru = cls(hidden, hidden, batch_first=True)
            self.post_gru = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU(0.1))
        else:
            # no temporal branch: the delay embedding alone drives the decoder
            self.bias = nn.Parameter(torch.zeros(1, hidden, 1, 1))
        self.snr_mlp = nn.Sequential(nn.Linear(1, snr_emb), nn.LeakyReLU(0.1))
        n_skip = hidden + snr_emb
        self.decoder = Decoder(enc_chs, n_skip, base_ch)

        if self.constraint == "output":
            # The estimate is *parameterised* in the delay domain: the model predicts K complex
            # delay coefficients from the pilot LS observation and reconstructs the channel as
            # F_K g, so the output is strictly confined to the K-dimensional delay subspace.  The
            # head is zero-initialised, so the model starts at the zero predictor and opens up
            # from there.  Note that an analytic least-squares projection is deliberately *not*
            # used: inverting the pilot-masked observation amplifies the noise by 1/sqrt(n_p) and
            # spreads it over the whole band, which inflates the estimate's power by an order of
            # magnitude.  Learning the map from observation to coefficients avoids that.
            n_p = len(range(0, n_fft, SYS_PILOT_SPACING))
            ctx_dim = hidden + snr_emb
            self.tap_head = nn.Sequential(nn.Linear(ctx_dim + 2 * n_p, 256), nn.LeakyReLU(0.1),
                                          nn.Linear(256, 2 * n_taps))
            self.register_buffer("pilot_k", torch.as_tensor(
                np.arange(0, n_fft, SYS_PILOT_SPACING), dtype=torch.long))
            # partial DFT basis F_K, (N_f, L_d); fixed, no parameters
            k = torch.arange(n_fft, dtype=torch.float32)
            d = torch.arange(n_taps, dtype=torch.float32)
            B = torch.exp(-2j * torch.pi * torch.outer(k, d) / n_fft) / (n_fft ** 0.5)
            self.register_buffer("basis_re", B.real.contiguous())
            self.register_buffer("basis_im", B.imag.contiguous())
            nn.init.zeros_(self.tap_head[2].weight)
            nn.init.zeros_(self.tap_head[2].bias)
            nn.init.zeros_(self.decoder.head.weight)
            nn.init.zeros_(self.decoder.head.bias)
            # learnable gate on the (unconstrained) full-resolution residual branch; it starts
            # nearly closed so the model begins as a pure delay-domain estimator.
            self.res_scale = nn.Parameter(torch.tensor(-4.0))

    # ------------------------------------------------------------------ helpers
    def _reconstruct(self, g: torch.Tensor) -> torch.Tensor:
        """Map delay coefficients (B, 2, L_d, N_t) to the channel grid (B, 2, N_f, N_t)."""
        gr, gi = g[:, 0], g[:, 1]                              # (B, L_d, N_t)
        br, bi = self.basis_re, self.basis_im                  # (N_f, L_d)
        h_re = torch.einsum("fl,blt->bft", br, gr) - torch.einsum("fl,blt->bft", bi, gi)
        h_im = torch.einsum("fl,blt->bft", br, gi) + torch.einsum("fl,blt->bft", bi, gr)
        return torch.stack([h_re, h_im], dim=1)

    def forward(self, x: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """x : (B, M, 4, N_f, N_t), snr_db : (B,) -> estimate (B, 2, N_f, N_t)."""
        b, m, c, nf, nt = x.shape
        feats_last = None
        pooled = []
        for i in range(m):
            xi = x[:, i]
            feats = self.encoder(xi)
            feats_last = feats
            pooled.append(feats[-1].mean(dim=(2, 3)))
        if self.use_temporal:
            seq = torch.stack(pooled, dim=1)                 # (B, M, C)
            out, _ = self.gru(self.pre_gru(seq))
            ctx_t = self.post_gru(out[:, -1]).view(b, -1, 1, 1)  # (B, hidden, 1, 1)
        else:
            ctx_t = self.bias.expand(b, -1, -1, -1)          # (B, hidden, 1, 1)
        if self.delay is not None:
            ctx_d = self.delay(x[:, -1])                     # (B, hidden, 1, 1)
        else:
            ctx_d = 0.0
        ctx = ctx_t + ctx_d
        ctx = torch.cat([ctx, self.snr_mlp(snr_db.view(b, 1)).view(b, -1, 1, 1)], dim=1)
        res = self.decoder(feats_last, ctx)
        if res.shape[-2:] != (nf, nt):
            res = F.interpolate(res, size=(nf, nt), mode="bilinear", align_corners=False)
        if self.constraint != "output":
            return res

        # ---- output-domain constraint: the estimate is a delay-domain model plus a residual
        # pilot observations of the slot being estimated: Re/Im of the LS values on the comb
        zr = x[:, -1, 0] * x[:, -1, 2]                        # (B, N_f, N_t)
        zi = x[:, -1, 1] * x[:, -1, 2]
        obs = torch.cat([zr[:, self.pilot_k, :], zi[:, self.pilot_k, :]], dim=1)  # (B, 2*n_p, N_t)
        ctx_t = ctx.flatten(1).unsqueeze(-1).expand(-1, -1, nt)                   # (B, C, N_t)
        g = self.tap_head(torch.cat([ctx_t, obs], dim=1).permute(0, 2, 1))        # (B, N_t, 2*L_d)
        g = g.permute(0, 2, 1).reshape(b, 2, self.delay.n_taps, nt)
        h0 = self._reconstruct(g)
        return h0 + torch.sigmoid(self.res_scale) * res


# ---------------------------------------------------------------------------------------
# deep-learning baselines
# ---------------------------------------------------------------------------------------
class ChannelNet(nn.Module):
    """SRCNN-style fully-convolutional estimator (ChannelNet, Solanki et al. 2018).

    Three convolutional layers on the low-resolution LS image, upsampled by bicubic
    interpolation beforehand / afterwards.  Single-slot, i.e. no temporal information.
    """

    def __init__(self, n_fft: int = 72, n_sym: int = 14, n_pilot_sym: int = 2, width: int = 64):
        super().__init__()
        self.n_fft, self.n_sym, self.n_pilot_sym = n_fft, n_sym, n_pilot_sym
        self.net = nn.Sequential(
            nn.Conv2d(2, width, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(width, width // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, 2, 5, padding=2),
        )

    def forward(self, x):
        b, m, c, nf, nt = x.shape
        img = x[:, -1, 0:2]                                # LS Re/Im (single slot)
        img = F.interpolate(img, size=(nf, nt), mode="bilinear", align_corners=False)
        return self.net(img)


class CDRN(nn.Module):
    """Compact residual CNN estimator with a single GRU for temporal aggregation.

    This is the closest published competitor: a residual convolutional denoiser applied to
    the LS estimate, here augmented with the same GRU temporal fusion so that the comparison
    isolates the contribution of the *delay-domain sparsity branch*.
    """

    def __init__(self, n_fft: int = 72, n_sym: int = 14, n_pilot_sym: int = 2,
                 width: int = 64, depth: int = 8, seq_len: int = 4, hidden: int = 128):
        super().__init__()
        layers = [nn.Conv2d(2, width, 3, padding=1, padding_mode="replicate")]
        for _ in range(depth):
            layers += [nn.ReLU(inplace=True),
                       nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate")]
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        self.gru = nn.GRU(width, hidden, batch_first=True) if seq_len > 1 else None
        self.post = nn.Linear(hidden, width) if seq_len > 1 else None

    def forward(self, x):
        b, m, c, nf, nt = x.shape
        feats = []
        for i in range(m):
            feats.append(self.body(x[:, i, 0:2]))     # LS Re/Im observation only
        h = feats[-1]
        if self.gru is not None:
            seq = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)
            g, _ = self.gru(seq)
            h = h + self.post(g[:, -1]).view(b, -1, 1, 1)
        return self.head(h)


class ResCNN(nn.Module):
    """A residual CNN estimator on the LS plane — the generic deep baseline.

    This is the most common learned-baseline family in the channel-estimation literature
    (residual correction applied to a noisy LS estimate, e.g. the CDRN-style denoising block of
    [6]).  It is deliberately given every advantage the proposed model has *except* the fixed
    delay-domain branch: it sees the same four input planes, it may aggregate the same ``M``
    slots through the same GRU, and it is trained with the same protocol.  Two configurations
    are used in the paper:

    * ``with_rnn=False`` — a single-slot residual CNN, a strong reference for what pure
      spatial capacity achieves.
    * ``with_rnn=True``  — the same network plus cross-slot temporal aggregation, which makes
      the comparison against the proposed model isolate the delay-domain branch.

    A strided first layer keeps the cost manageable at ``N_f = 256``.
    """

    def __init__(self, n_fft: int = 256, n_sym: int = 14, seq_len: int = 4,
                 width: int = 64, n_blocks: int = 3, hidden: int = 128,
                 with_rnn: bool = False, in_ch: int = 4):
        super().__init__()
        self.seq_len, self.with_rnn = seq_len, with_rnn
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, stride=2, padding=1, padding_mode="replicate", bias=False),
            nn.GroupNorm(min(8, width), width), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate", bias=False),
                nn.GroupNorm(min(8, width), width), nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate", bias=False),
                nn.GroupNorm(min(8, width), width))
            for _ in range(n_blocks)])
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, 2, 3, padding=1))
        if with_rnn and seq_len > 1:
            self.pre = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(inplace=True))
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.post = nn.Linear(hidden, width)

    def forward(self, x, snr_db=None):
        b, m, c, nf, nt = x.shape
        feats = []
        for i in range(m):
            s = self.stem(x[:, i])
            h = self.act(self.blocks(s) + s)
            feats.append(h)
        h = feats[-1]
        if self.with_rnn and m > 1:
            seq = torch.stack([f.mean(dim=(2, 3)) for f in feats], dim=1)
            g, _ = self.gru(self.pre(seq))
            h = h + self.post(g[:, -1]).view(b, -1, 1, 1)
        out = self.up(h)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for m in (1, 4):
        net = TDFNet(n_fft=256, n_sym=14, n_pilot_sym=4, seq_len=m)
        x = torch.randn(2, m, 4, 256, 14)
        s = torch.rand(2) * 30 - 5
        y = net(x, s)
        print("TDFNet M=%d  params=%d  out=%s" % (m, count_parameters(net), tuple(y.shape)))
    for name, net in (("ChannelNet", ChannelNet(n_fft=256, n_sym=14, n_pilot_sym=4),
                       ), ("CDRN", CDRN(n_fft=256, n_sym=14, n_pilot_sym=4))):
        x = torch.randn(2, 4, 4, 256, 14)
        print("%-11s params=%d out=%s" % (name, count_parameters(net), tuple(net(x).shape)))
