"""Export the learned delay-domain window and the true delay profile (for Fig. 5)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from config import SYS, EVAL, ChannelConfig
from channel import TDLChannel
from eval import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

device = torch.device("cpu")
net, ck = load_model("tdfnet_M4", device)
w = net.delay.window.detach().numpy().ravel()
print("learned window (L_d=%d):" % len(w))
print(np.round(w, 4))

rng = np.random.default_rng(0)
ch = TDLChannel(ChannelConfig(profile="TDL-C", delay_spread_ns=EVAL.delay_spread_ns,
                              velocity_kmh=100.0), rng)
taps = ch.tap_power_d[ch.tap_ids]
print("\ntrue discrete tap power:", np.round(taps, 4), "at delays", ch.tap_ids)
print("L_cp = %d samples, L_d = %d" % (int(round(SYS.n_fft * 0.0703)), len(w)))

out = {"window": w.tolist(),
       "active_taps": [ch.tap_ids.tolist(), taps.tolist()],
       "L_cp": int(round(SYS.n_fft * 0.0703)),
       "L_d": int(len(w)),
       "params": ck["n_params"]}
with open(os.path.join(RESULTS, "delay_window.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\n[saved] results/delay_window.json")
