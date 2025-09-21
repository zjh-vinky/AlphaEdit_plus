import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from util.globals import *


class AlphasetDataset:


    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, *args, **kwargs):

        data_dir = Path("AlphaSet")
        alpha_loc = data_dir / "K0.json"

        if not alpha_loc.exists():
            print(f"{alpha_loc} does not exist. ")

        with open(alpha_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]

            if record["subject"] not in record["src"]:
                print(f"[WARN] subject '{record['subject']}' not found in src: '{record['src']}' (case_id={i})")

            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
