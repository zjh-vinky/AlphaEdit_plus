# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import typing
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast
from util.globals import *
from rome.layer_stats import layer_stats

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words
from .AlphaEditPlus_hparams import AlphaEditPlusHyperParams
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_new_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_alphaset import compute_rewrite_quality_alphaset

# Caches
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# Keep previous keys for building Kp / Lambda_p per layer across calls
_PREV_KEYS: Dict[int, torch.Tensor] = {}     # layer -> [h, M], columns are keys


def apply_AlphaEditPlus_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditPlusHyperParams,
    record_chunks,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None, 
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = next(model.parameters()).device
    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1) #v1

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        K1 = compute_ks(model, tok, requests, hparams, layer, context_templates).T  # [h, N]
        print(f"Writing {K1.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error v0
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[r["prompt"] for r in requests],
            words=[r["subject"] for r in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T  # [d, N]

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        
        targets = zs - cur_zs  # [d,N]
        # print(targets)  
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        repeat_factor = (K1.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        R0 = targets / (len(hparams.layers) - i) # [d,N]

        Kp = _PREV_KEYS.get(layer, None)

        if Kp is None:
            Lambda_p = None
        else:
            Lambda_p = build_lambda_p(Kp, K1)  # [M], diag later

        # --- Base projector P ---
        P_base = P[i, :, :].cuda()  # [h,h]

        # --- Build candidates from covariance eigenvectors (ascending eigenvalues) ---
        eigvecs, eigvals, _, mask = _eigen_pool_for_layer(model, tok, layer,hparams)
        order = torch.argsort(eigvals)  # ascending
        cand_idx = [int(i) for i in order.tolist() if not bool(mask[i])]

        # --- Greedy expansion of \tilde{P} ---
        P_tilde = torch.zeros_like(P_base, device=device)
        P_mod = P_base + P_tilde  # [h,h]

        # initial solve and objective
        upd_matrix = solve_delta_closed_form(
                P[i, :, :].cuda(), K1, R0, i, hparams, Kp=Kp, Lambda_p=Lambda_p
            ).T

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        J_prev = objective_value(upd_matrix, K1, R0, i, hparams.beta, hparams.L2, Kp=Kp, Lambda_p=Lambda_p)

        added = 0
        batch_size = 10

        print("cand_idx:", len(cand_idx))

        while added < len(cand_idx):
            batch = cand_idx[added:added+batch_size]
            if not batch:
                print(f"[AlphaEdit+] No more candidates at {added}")
                break
            P_tilde_try = P_tilde

            U_batch = eigvecs[:, batch].to(device)  # [h, b]
            P_tilde_try = P_tilde + U_batch @ U_batch.T  # [h, h]

            P_mod_try = P_base + P_tilde_try

            upd_matrix_try = solve_delta_closed_form(
                P_mod_try, K1, R0, i, hparams, Kp=Kp, Lambda_p=Lambda_p
            ).T

            J_new = objective_value(upd_matrix_try, K1, R0, i, hparams.beta, hparams.L2, Kp=Kp, Lambda_p=Lambda_p)

            improved = ((J_prev - J_new) / J_prev) > hparams.nullspace_searchthreshold * batch_size 
            print(f"J_prev={J_prev:.6f}, J_new={J_new:.6f}, rel_diff={(J_prev-J_new)/J_prev:.6f} -> {'ACCEPT' if improved else 'REJECT'}")
            if not improved:
                print(f"[AlphaEdit+] Stopping at {added} new singular vectors")
                break

            # accept batch
            P_tilde = P_tilde_try
            P_mod = P_mod_try
            upd_matrix = upd_matrix_try
            J_prev = J_new
            added += len(batch)

        print(f"[AlphaEdit+] Total new singular vectors added: {added}")
        if added >= len(cand_idx):
            print(f"[AlphaEdit+] No more candidates at {added}, cand_idx: {len(cand_idx)}")


        T = 10
        tau_r = hparams.r

        targets = zs - cur_zs                      # [d, N0]
        r_norm = torch.linalg.norm(targets, dim=0) # [N0]
        print(f"r_norm: {r_norm}")

        repeat_factor = (K1.size(1) // targets.size(1))

        if torch.all(r_norm <= tau_r):
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            R0 = targets / (len(hparams.layers) - i)  # Distribute residual across layers
            R0_list = [R0] 
        else:
            R0_list = []
            mask = (r_norm > tau_r).float()            # [N0]
            mask = mask.repeat_interleave(repeat_factor)  # [N]

            for t in range(T+1):                   
                beta_scalar = (T - t) / (2 * T)     
                # beta_scalar = t / T    
                beta = beta_scalar * mask              # [N]
                v_t = zs + beta.unsqueeze(0) * (cur_zs - zs)   # [d, N]

                targets_t = v_t - cur_zs                     # [d, N]
                R0 = targets_t / (len(hparams.layers) - i)   # [d, N]
                R0_list.append(R0) 

        Kp_term = 0
        if Kp is not None:
            Kp = Kp.to(K1.device)
            # Kp: [h, M], Lambda_p: [M]
            Kp_term = Kp * Lambda_p.unsqueeze(0)  # [h, M]
            Kp_term = Kp_term @ Kp.T              # [h, h]

        A = P_mod.cuda() @ (K1 @ K1.T + hparams.beta * (i+1) * Kp_term) + hparams.L2 * torch.eye(K1.shape[0], dtype=torch.float,device="cuda")

        B_list = [P_mod.cuda() @ K1 @ R0.T for R0 in R0_list]  # [h, d]
        B = torch.cat(B_list, dim=1)

        upd_matrix_all = torch.linalg.solve(A, B)

        upd_matrix_list = torch.split(upd_matrix_all, B.shape[1] // len(R0_list), dim=1) 

        orig_weight = weights[weight_name].clone()
        acc_list = []
        for idx, upd_matrix in enumerate(upd_matrix_list):
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            with torch.no_grad():
                weights[weight_name][...] = orig_weight + upd_matrix

            val_rewrite_prompts_correct = []
            val_paraphrase_prompts_correct = []
            for record in record_chunks:
                ret = compute_rewrite_quality_zsre(
                    model,
                    tok,
                    record,
                    None,
                    None,
                )
                # ret = compute_rewrite_quality_counterfact(
                #     model,
                #     tok,
                #     record,
                #     None,
                #     None,
                # )
                val_rewrite_prompts = ret.get("rewrite_prompts_correct", [])
                val_paraphrase_prompts = ret.get("paraphrase_prompts_correct", [])
                if isinstance(val_rewrite_prompts, list):
                    val_rewrite_prompts_correct.extend(val_rewrite_prompts)
                else:
                    val_rewrite_prompts_correct.append(val_rewrite_prompts)

                if isinstance(val_paraphrase_prompts, list):
                    val_paraphrase_prompts_correct.extend(val_paraphrase_prompts)
                else:
                    val_paraphrase_prompts_correct.append(val_paraphrase_prompts)

            val_rewrite_prompts_acc = sum(bool(x) for x in val_rewrite_prompts_correct) / len(val_rewrite_prompts_correct)
            val_paraphrase_prompts_acc = sum(bool(x) for x in val_paraphrase_prompts_correct) / len(val_paraphrase_prompts_correct)
            acc_list.append(val_rewrite_prompts_acc + val_paraphrase_prompts_acc)
            print(f"idx: {idx}, acc: {val_rewrite_prompts_acc:.8f}, {val_paraphrase_prompts_acc:.8f}")

            with torch.no_grad():
                weights[weight_name][...] = orig_weight


        best_idx = int(np.argmax(acc_list))
        print(f"upd_matrix: {best_idx}, acc={acc_list[best_idx]:.8f}")
        upd_matrix = upd_matrix_list[best_idx]


        # ---- Apply update: W <- W + Delta ----
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # ---- Accumulate keys for future Lambda_p ----
        _accumulate_prev_keys(layer, K1.detach().to("cpu"))
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Same contract as AlphaEdit: return (inverse) covariance on GPU.
    """
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    print(f"[AlphaEdit+] Retrieving covariance for {model_name} @ {layer_name}")

    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    cov = COV_CACHE[key].to(next(model.parameters()).device)
    return torch.inverse(cov) if inv else cov


# ==========================

def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [[
            f.replace("{", " ").replace("}", " ") + ". {}"
            for f in generate_fast(model, tok,
                                   ["The", "Therefore", "Because", "I", "You"],
                                   n_gen_per_prompt=n_gen // 5,
                                   max_out_len=length)
        ] for length, n_gen in [(10, 5)]]
        print(f"[AlphaEdit+] Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE


def _eigen_pool_for_layer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    hparams: AlphaEditPlusHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (eigvecs, eigvals, base_U_small) for the layer from covariance.
    eigvecs: [h, h], eigvals: [h] (ascending)
    base_U_small: [h, r] eigenvectors used in base P (thresholded by nullspace_threshold)
    """
    force_recompute = False
    cov = get_cov(
        model, 
        tok, 
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10, 
        hparams.mom2_dtype, 
        force_recompute=False
    ).detach().to("cpu")
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    mask = (S < hparams.nullspace_threshold)
    base_U_small = U[:, mask]               # columns used in P
    return U, S, base_U_small, mask


def solve_delta_closed_form(
    P: torch.Tensor,      # [h,h]
    K1: torch.Tensor,         # [h,N]
    R0: torch.Tensor,         # [d,N]
    i,
    hparams: AlphaEditPlusHyperParams,
    *,
    Kp: Optional[torch.Tensor] = None,    # [h,M]
    Lambda_p: Optional[torch.Tensor] = None,  # [M]
) -> torch.Tensor:

    Kp_term = 0
    if Kp is not None:
        Kp = Kp.to(K1.device)
        # Kp: [h, M], Lambda_p: [M]
        Kp_term = Kp * Lambda_p.unsqueeze(0)  # [h, M] 
        Kp_term = Kp_term @ Kp.T              # [h, h]

    upd_matrix = torch.linalg.solve(
                P.cuda() @ (K1 @ K1.T + hparams.beta * (i+1) * Kp_term) + hparams.L2 * torch.eye(K1.shape[0], dtype=torch.float,device="cuda"),
                P.cuda() @ K1 @ R0.T
            )
    
    return upd_matrix


def objective_value(
    upd_matrix: torch.Tensor,      # [d,h]
    K1: torch.Tensor,         # [h,N]
    R0: torch.Tensor,         # [d,N]
    i,
    beta: float,
    lam_obj: float,
    *,
    Kp: Optional[torch.Tensor] = None,     # [h,M]
    Lambda_p: Optional[torch.Tensor] = None,  # [M]
) -> float:
    """
    J = || Delta P_mod K1 - R0 ||_F^2
      + || Delta P_mod Kp Λ_p^{1/2} ||_F^2
      + lam_obj * || Delta P_mod ||_F^2
    """
    device = K1.device
    DP = upd_matrix
    term1 = torch.linalg.norm(DP @ K1 - R0).pow(2)
    term2 = torch.tensor(0.0, device=device)
    if Kp is not None and Kp.numel() > 0 and Lambda_p is not None:
        Kp = Kp.to(device)
        Lhalf = torch.sqrt(torch.clamp(Lambda_p.to(device), min=0.0))
        term2 = torch.linalg.norm(DP @ (Kp * Lhalf.unsqueeze(0))).pow(2)
    term3 = lam_obj * torch.linalg.norm(DP).pow(2)
    total = term1 + beta * (i+1) * term2 + term3
    return float(total.item())


def build_lambda_p(Kp: torch.Tensor, K1: torch.Tensor) -> torch.Tensor:
    """
    Lambda_p[j] = 1 - max_i |cos( k_pj, k_1i )|
    k_pj \in Kp (previous), k_1i \in K1 (current).
    Shapes: Kp [h,M], K1 [h,N]
    """
    device = K1.device
    # L2 normalize columns
    def _norm_cols(X):
        nrm = torch.clamp(torch.linalg.norm(X, dim=0, keepdim=True), min=1e-12)
        return X / nrm

    Kp_n = _norm_cols(Kp.to(device))  # [h,M]
    K1_n = _norm_cols(K1.to(device))  # [h,N]
    sims = torch.abs(Kp_n.T @ K1_n)   # [M,N]
    smax, _ = sims.max(dim=1)         # [M]
    Lambda = 1.0 - smax

    high_sim_idx = (sims > 0.8).nonzero(as_tuple=False)
    if high_sim_idx.numel() > 0:
        print(f"[build_lambda_p] High similarity pairs (Kp_idx, K1_idx): {high_sim_idx.tolist()}")

    print(f"[build_lambda_p] Lambda_p weights: {Lambda.cpu().numpy().round(4).tolist()}")

    return Lambda.detach()


def _accumulate_prev_keys(layer: int, K1_cpu: torch.Tensor):
    """
    Append current K1 (columns) into global _PREV_KEYS for given layer (CPU).
    """
    global _PREV_KEYS
    if layer not in _PREV_KEYS or _PREV_KEYS[layer].numel() == 0:
        _PREV_KEYS[layer] = K1_cpu.clone()
    else:
        old = _PREV_KEYS[layer]
        cat = torch.cat([old, K1_cpu], dim=1)
        _PREV_KEYS[layer] = cat  


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Same helper as in AlphaEdit: transpose if necessary to match weight shape.
    """
    if isinstance(matrix, (list, tuple)):
        return [upd_matrix_match_shape(m, shape) for m in matrix]
    elif isinstance(matrix, torch.Tensor) and matrix.ndim == 3:
        # batched tensor: [batch, d, h] 或 [batch, h, d]
        if matrix.shape[1:] == shape:
            return matrix
        elif matrix.shape[0] == shape[0] and matrix.shape[2] == shape[1]:
            return matrix.transpose(1, 2)
        elif matrix.shape[1] == shape[1] and matrix.shape[2] == shape[0]:
            return matrix.transpose(1, 2)
        else:
            raise ValueError(
                f"[AlphaEdit+] Batched update matrix shape {matrix.shape} not matching weight shape {shape}."
            )
    else:
        if matrix.shape == shape:
            return matrix
        elif matrix.T.shape == shape:
            return matrix.T
        else:
            raise ValueError(
                f"[AlphaEdit+] Update matrix shape {matrix.shape} not matching weight shape {shape}."
            )


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        if 'llama' in model.config._name_or_path.lower():
            correct_id = correct_id[:, 1].squeeze()
        else:
            correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()