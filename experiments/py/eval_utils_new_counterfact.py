"""
Modified version of eval_utils_counterfact.py using eval_utils_zsre.py's evaluation approach.
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.
    
    Modified to use zsre-style evaluation approach.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]

    # === 使用 zsre 风格的评估方式 ===
    
    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    
    # Flatten all the evaluated prefixes into one list.
    target_tok = tok(" " + target_new["str"])["input_ids"]
    if 'llama' in model.config._name_or_path.lower():
        target_tok = target_tok[1:]
    
    inp_prompts_og = list(chain(*prob_prompts))
    inp_prompts = [
        el + tok.decode(target_tok[:i]) if 'llama' not in model.config._name_or_path.lower() or i == 0 
        else el + ' ' + tok.decode(target_tok[:i])
        for el in inp_prompts_og
        for i in range(len(target_tok))
    ]
    inp_targets = [
        tok.decode(target_tok[i])
        for _ in range(len(inp_prompts_og))
        for i in range(len(target_tok))
    ]

    # Test batch prediction accuracy using zsre approach
    stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

    # === 为neighborhood prompts创建逐token评估 ===
    neighborhood_prompts_expanded = []
    neighborhood_targets_expanded = []
    
    # 对每个neighborhood prompt进行逐token展开
    for neighborhood_prompt in neighborhood_prompts:
        # tokenize target_true来获取逐token的目标
        target_true_tok = tok(" " + target_true["str"])["input_ids"]
        if 'llama' in model.config._name_or_path.lower():
            target_true_tok = target_true_tok[1:]  # 跳过BOS token
        
        # 为每个token位置创建prompt和target
        for i in range(len(target_true_tok)):
            # 构造增量prompt (类似zsre的方式)
            if i == 0:
                expanded_prompt = neighborhood_prompt
            else:
                partial_answer = tok.decode(target_true_tok[:i])
                if 'llama' not in model.config._name_or_path.lower():
                    expanded_prompt = neighborhood_prompt + partial_answer
                else:
                    expanded_prompt = neighborhood_prompt + ' ' + partial_answer
            
            # 当前位置应该预测的token
            target_token = tok.decode(target_true_tok[i])
            
            neighborhood_prompts_expanded.append(expanded_prompt)
            neighborhood_targets_expanded.append(target_token)

    # 使用展开后的prompts进行评估
    neighborhood_correct = test_batch_prediction_acc(
        model, 
        tok, 
        neighborhood_prompts_expanded, 
        neighborhood_targets_expanded
    )

    # 将结果按照原始neighborhood prompts的数量重新分组
    tokens_per_prompt = len(target_true_tok)
    neighborhood_results_grouped = [
        neighborhood_correct[i*tokens_per_prompt:(i+1)*tokens_per_prompt] 
        for i in range(len(neighborhood_prompts))
    ]
    
    # 计算每个neighborhood prompt的整体准确性（所有token都对才算对）
    neighborhood_overall_correct = [
        all(group) for group in neighborhood_results_grouped
    ]

    # 合并所有结果
    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(
        [l * len(target_tok) for l in map(len, prob_prompts)]
    ).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    
    # Structure the results as a dictionary (zsre style).
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }

    ret["neighborhood_prompts_correct"] = neighborhood_overall_correct

    # === 保留原有的生成评估部分 ===
    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction_acc(model, tok, prompts: typing.List[str], target):
    """
    Test batch prediction accuracy using zsre approach.
    This function is copied from eval_utils_zsre.py
    """
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


# === 保留原有的生成评估函数 ===
def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    # encs = vec.transform([text_a, text_b]).A
    encs = vec.transform([text_a, text_b]).toarray()
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()