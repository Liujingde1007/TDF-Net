"""Training driver.

Loss
----
We minimise the *normalised* MSE, i.e. the per-sample error power relative to the per-sample
channel power.  Using a per-sample normalisation instead of a global one matters: without it
the gradient is dominated by fading peaks (deep fades are well estimated in absolute terms
but badly estimated in relative terms) and the model under-fits low-power realisations.

    L = mean_over_batch [ ||H_hat - H||^2 / ||H||^2 ]

Checkpoints store the full configuration so that every reported number can be reproduced.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import SYS, TRAIN, ChannelConfig
from data import ChannelSequenceDataset, make_sequences
from models import TDFNet, ChannelNet, CDRN, ResCNN, count_parameters

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def build_model(name: str, seq_len: int, base_ch: int | None = None,
                n_taps: int | None = None, hidden: int = 128,
                use_delay: bool = True, rnn: str = "gru",
                with_rnn: bool = False, constraint: str = "feature") -> nn.Module:
    if name == "tdfnet":
        return TDFNet(n_fft=SYS.n_fft, n_sym=SYS.n_symbols,
                      n_pilot_sym=len(SYS.pilot_symbols),
                      n_taps=n_taps or TRAIN.n_delay_taps,
                      base_ch=base_ch or TRAIN.base_ch, seq_len=seq_len, hidden=hidden,
                      use_delay=use_delay, rnn=rnn, constraint=constraint)
    if name == "channelnet":
        return ChannelNet(n_fft=SYS.n_fft, n_sym=SYS.n_symbols,
                          n_pilot_sym=len(SYS.pilot_symbols))
    if name == "cdrn":
        return CDRN(n_fft=SYS.n_fft, n_sym=SYS.n_symbols,
                    n_pilot_sym=len(SYS.pilot_symbols), seq_len=seq_len)
    if name == "rescnn":
        # infer the recurrent variant from the argument or from a seq_len > 1 window
        return ResCNN(n_fft=SYS.n_fft, n_sym=SYS.n_symbols, seq_len=seq_len,
                      with_rnn=with_rnn, width=64, n_blocks=3)
    raise ValueError("unknown model: %s" % name)


def nmse_loss(H_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Per-sample normalised MSE."""
    num = ((H_hat - H) ** 2).sum(dim=(1, 2, 3))
    den = (H ** 2).sum(dim=(1, 2, 3)) + 1e-12
    return (num / den).mean()


def evaluate(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, y, snr in loader:
            x, y, snr = x.to(device), y.to(device), snr.to(device)
            out = model(x, snr) if isinstance(model, TDFNet) else model(x)
            tot += nmse_loss(out, y).item() * x.shape[0]
            n += x.shape[0]
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tdfnet",
                    choices=["tdfnet", "channelnet", "cdrn", "rescnn"])
    ap.add_argument("--seq-len", type=int, default=TRAIN.seq_len)
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--n-train", type=int, default=TRAIN.n_train)
    ap.add_argument("--n-val", type=int, default=TRAIN.n_val)
    ap.add_argument("--n-taps", type=int, default=TRAIN.n_delay_taps)
    ap.add_argument("--base-ch", type=int, default=TRAIN.base_ch)
    ap.add_argument("--tag", default=None, help="checkpoint tag (default: model_seqlen)")
    ap.add_argument("--seed", type=int, default=TRAIN.seed)
    ap.add_argument("--no-delay", action="store_true",
                    help="ablation: remove the fixed delay-domain (IDFT) sparsity branch")
    ap.add_argument("--rnn", default="gru", choices=["gru", "lstm"],
                    help="temporal aggregation cell")
    ap.add_argument("--with-rnn", action="store_true",
                    help="ResCNN baseline: add cross-slot GRU aggregation")
    ap.add_argument("--constraint", default="feature", choices=["feature", "output"],
                    help="how the delay prior enters: as a feature context, or as a constraint "
                         "on the output (delay-domain model + learned residual)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)

    tag = args.tag or "%s_M%d" % (args.model, args.seq_len)
    device = torch.device("cpu")

    # ------------------------------------------------------------------ data
    rng = np.random.default_rng(args.seed)
    print("[data] generating %d training sequences ..." % args.n_train, flush=True)
    t0 = time.time()
    Xtr, Ytr, Mtr = make_sequences(args.n_train, args.seq_len, rng,
                                   TRAIN.velocity_group, TRAIN.snr_train_db, progress=True)
    print("[data] training set ready in %.1f s" % (time.time() - t0), flush=True)
    Xva, Yva, Mva = make_sequences(args.n_val, args.seq_len, rng,
                                   TRAIN.velocity_group, TRAIN.snr_train_db)
    tr = DataLoader(ChannelSequenceDataset(Xtr, Ytr, Mtr), batch_size=args.batch_size,
                    shuffle=True, drop_last=True, num_workers=0)
    va = DataLoader(ChannelSequenceDataset(Xva, Yva, Mva), batch_size=args.batch_size,
                    shuffle=False, num_workers=0)

    # ------------------------------------------------------------------ model
    model = build_model(args.model, args.seq_len, base_ch=args.base_ch, n_taps=args.n_taps,
                        use_delay=not args.no_delay, rnn=args.rnn, with_rnn=args.with_rnn,
                        constraint=args.constraint)
    n_par = count_parameters(model)
    print("[model] %s  trainable parameters = %d" % (args.model, n_par), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=TRAIN.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)

    history = {"tag": tag, "model": args.model, "seq_len": args.seq_len,
               "n_params": n_par, "epochs": [], "train_loss": [], "val_loss": [],
               "lr": [], "config": {"n_train": args.n_train, "n_val": args.n_val,
                                    "batch_size": args.batch_size, "lr0": args.lr,
                                    "n_taps": args.n_taps, "base_ch": args.base_ch,
                                    "use_delay": not args.no_delay, "rnn": args.rnn,
                                    "seed": args.seed, **TRAIN.as_dict()}}
    best = float("inf")
    ckpt_path = os.path.join(CKPT, tag + ".pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run, n = 0.0, 0
        for x, y, snr in tr:
            opt.zero_grad(set_to_none=True)
            out = model(x, snr) if isinstance(model, TDFNet) else model(x)
            loss = nmse_loss(out, y)
            loss.backward()
            if TRAIN.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
            opt.step()
            run += loss.item() * x.shape[0]
            n += x.shape[0]
        tr_loss = run / max(n, 1)
        va_loss = evaluate(model, va, device)
        sched.step()

        history["epochs"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        marker = ""
        if va_loss < best:
            best = va_loss
            torch.save({"model": args.model, "seq_len": args.seq_len, "state_dict": model.state_dict(),
                        "n_params": n_par, "val_loss": va_loss, "epoch": ep,
                        "use_delay": not args.no_delay, "rnn": args.rnn,
                        "with_rnn": args.with_rnn, "constraint": args.constraint,
                        "base_ch": args.base_ch, "n_taps": args.n_taps,
                        "args": vars(args), "config": history["config"]}, ckpt_path)
            marker = "  *saved"
        print("[ep %3d/%3d] train %.5f (%.2f dB)  val %.5f (%.2f dB)  lr %.2e  %.1fs%s"
              % (ep, args.epochs, tr_loss, 10 * np.log10(tr_loss),
                 va_loss, 10 * np.log10(va_loss), opt.param_groups[0]["lr"],
                 time.time() - t0, marker), flush=True)

    history["best_val_loss"] = best
    history["best_val_nmse_db"] = 10 * np.log10(best)
    with open(os.path.join(RESULTS, "history_%s.json" % tag), "w") as f:
        json.dump(history, f, indent=2)
    print("[done] best val NMSE = %.2f dB -> %s" % (10 * np.log10(best), ckpt_path))


if __name__ == "__main__":
    main()
