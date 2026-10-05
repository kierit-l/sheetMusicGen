"""Which hand plays each performed note, with the PM2S hand-part model.

The networks are from "Performance MIDI-to-Score Conversion by Neural Beat
Tracking" (L. Liu, Q. Kong, V. Morfi and E. Benetos, ISMIR 2022),
https://github.com/cheriell/PM2S, trained on ASAP, A-MAPS and CPM. Only the
inference code is reproduced here; the weights (~20 MB each) download from
Zenodo on first use.

MIT License, Copyright (c) 2022 Lele Liu. Permission is hereby granted, free
of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to
the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""

from __future__ import annotations

import functools
import urllib.request
from pathlib import Path

import numpy as np

from .transcribe import Note

WEIGHTS_URL = "https://zenodo.org/records/10520196/files/{}.pth?download=1"
WEIGHTS_DIR = Path.home() / ".cache" / "sheetmusicgen" / "pm2s"
RESOLUTION = 0.01  # seconds per step of the onset-shift encoding
MAX_SHIFT = 4.0  # seconds; longer gaps between onsets are encoded as this


def _network(out_features: int, activation: str):
    import torch.nn as nn

    in_features = 128 + int(MAX_SHIFT / RESOLUTION) + 1 + 2
    hidden, kernel = 512, 9

    def conv(cin, cout, width):
        return [
            nn.Conv2d(cin, cout, kernel_size=(kernel, width), padding=(kernel // 2, 0)),
            nn.BatchNorm2d(cout),
            nn.ELU(),
            nn.Dropout(0.15),
        ]

    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                *conv(1, hidden // 4, in_features), *conv(hidden // 4, hidden // 2, 1), *conv(hidden // 2, hidden, 1)
            )

        def forward(self, x):
            return self.conv(x.unsqueeze(1)).squeeze(3).transpose(1, 2)

    class GRUBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.grus_beat = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=0.15, bidirectional=True)
            self.linear = nn.Linear(hidden * 2, hidden)

        def forward(self, x):
            return self.linear(self.grus_beat(x)[0])

    class Output(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden, out_features)
            self.activation = nn.Sigmoid() if activation == "sigmoid" else nn.LogSoftmax(dim=2)

        def forward(self, x):
            return self.activation(self.linear(x))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs, self.gru, self.out = ConvBlock(), GRUBlock(), Output()

        def forward(self, x):
            return self.out(self.gru(self.convs(x)))

    return Model()


@functools.cache
def _load(name: str, out_features: int, activation: str):
    import torch

    path = WEIGHTS_DIR / f"{name}.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL.format(name), tmp)
        tmp.rename(path)
    model = _network(out_features, activation)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def _features(notes: list[Note]):
    """The models' input: per note, pitch, gap since the previous onset, duration and velocity."""
    import torch
    import torch.nn.functional as F

    onset = torch.tensor([n.onset for n in notes], dtype=torch.float32)
    shift = torch.clamp(torch.diff(onset, prepend=onset[:1]), 0, MAX_SHIFT)
    return torch.cat(
        [
            F.one_hot(torch.tensor([n.pitch for n in notes]), 128).float(),
            F.one_hot(torch.round(shift / RESOLUTION).long(), int(MAX_SHIFT / RESOLUTION) + 1).float(),
            torch.tensor([[n.offset - n.onset] for n in notes], dtype=torch.float32),
            torch.tensor([[n.velocity / 127] for n in notes], dtype=torch.float32),
        ],
        dim=1,
    ).unsqueeze(0)


def right_hand_probs(notes: list[Note]) -> np.ndarray:
    """For each of `notes` (sorted by onset), the probability that the right hand plays it."""
    import torch

    with torch.inference_mode():
        left = _load("RNNHandPartModel", 1, "sigmoid")(_features(notes))[0, :, 0].numpy()  # trained on 1 = left hand
    return 1 - left

