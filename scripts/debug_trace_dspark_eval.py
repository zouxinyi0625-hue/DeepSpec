from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from deepspec.data.parser import encode_chat_messages
from deepspec.eval.base_evaluator import (
    has_stop_token,
    resolve_stop_token_ids,
    verify_draft_tokens,
)
from deepspec.eval.dspark.draft_ops import build_dspark_proposal, forward_dspark_draft_block
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.utils.sampling import logits_to_probs, sample_from_probs


def print_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        print(f"{name}: None")
        return
    print(f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace one DSpark Gemma4 speculative-decoding sample step by step.")
    parser.add_argument("--target-name-or-path", default="google/gemma-4-12B-it")
    parser.add_argument("--draft-name-or-path", required=True)
    parser.add_argument("--prompt", default="Explain speculative decoding in one short paragraph.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print("=== Loading models ===")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_name_or_path,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device=device).eval()
    draft_model = Gemma4DSparkModel.from_pretrained(
        args.draft_name_or_path,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_name_or_path)
    stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)

    messages = [{"role": "user", "content": args.prompt}]
    input_ids = encode_chat_messages(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(device)
    print_tensor("input_ids", input_ids)
    print(f"prompt_text={args.prompt!r}")

    max_proposal_tokens = int(draft_model.block_size)
    max_length = input_ids.shape[1] + int(args.max_new_tokens)
    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    print("\n=== Target prefill ===")
    with torch.inference_mode():
        initial_output = target_model(
            input_ids=input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
            logits_to_keep=1,
        )
    output_ids[:, : input_ids.shape[1]] = input_ids
    first_token = sample_from_probs(logits_to_probs(initial_output.logits, float(args.temperature)))
    output_ids[:, input_ids.shape[1] : input_ids.shape[1] + 1] = first_token
    print_tensor("initial_output.logits", initial_output.logits)
    print(f"initial hidden_states count={len(initial_output.hidden_states)}")
    for layer_id in draft_model.target_layer_ids:
        idx = 0 if int(layer_id) == -1 else int(layer_id) + 1
        print_tensor(f"initial hidden_states[{idx}] for target_layer_id={layer_id}", initial_output.hidden_states[idx])
    print(f"first target token id={int(first_token[0,0].item())} text={tokenizer.decode(first_token[0].tolist())!r}")

    context = SimpleNamespace(
        past_key_values_draft=DynamicCache(),
        target_hidden_states=extract_context_feature(
            initial_output.hidden_states,
            draft_model.target_layer_ids,
        ),
    )
    print_tensor("context.target_hidden_states after prefill", context.target_hidden_states)

    start = input_ids.shape[1]
    accepted_total = 0
    round_idx = 0
    with torch.inference_mode():
        while start < max_length and round_idx < int(args.max_rounds):
            round_idx += 1
            print(f"\n=== Round {round_idx} start={start} ===")
            print(f"draft past seq len before={context.past_key_values_draft.get_seq_length()}")
            print_tensor("context.target_hidden_states before propose", context.target_hidden_states)

            draft_input_ids = torch.full(
                (output_ids.size(0), max_proposal_tokens),
                int(draft_model.mask_token_id),
                dtype=torch.long,
                device=output_ids.device,
            )
            draft_input_ids[:, 0] = output_ids[:, start]
            print_tensor("draft_input_ids", draft_input_ids)
            print(f"draft_input_text={tokenizer.decode(draft_input_ids[0].tolist())!r}")

            block_hidden = forward_dspark_draft_block(
                draft_model,
                draft_input_ids=draft_input_ids,
                position_ids=position_ids,
                past_key_values_draft=context.past_key_values_draft,
                target_hidden_states=context.target_hidden_states,
                start=start,
                block_size=max_proposal_tokens,
            )
            print_tensor("block_hidden", block_hidden)
            print(f"draft past seq len after forward/crop={context.past_key_values_draft.get_seq_length()}")

            proposal = build_dspark_proposal(
                model=draft_model,
                draft_input_ids=draft_input_ids,
                block_hidden=block_hidden,
                block_size=max_proposal_tokens,
                temperature=float(args.temperature),
                confidence_threshold=float(args.confidence_threshold),
            )
            print(f"proposal.draft_token_count={proposal.draft_token_count}")
            print_tensor("proposal.verify_input_ids", proposal.verify_input_ids)
            print(f"proposal.verify_text={tokenizer.decode(proposal.verify_input_ids[0].tolist())!r}")
            if proposal.confidence_logits is not None:
                print_tensor("proposal.confidence_logits", proposal.confidence_logits)
                print(f"proposal.confidence_probs={proposal.confidence_logits.sigmoid()[0].detach().cpu().tolist()}")

            verification = verify_draft_tokens(
                target_model=target_model,
                proposal=proposal,
                position_ids=position_ids,
                start=start,
                past_key_values_target=past_key_values_target,
                temperature=float(args.temperature),
                max_proposal_tokens=max_proposal_tokens,
                current_token_ids=output_ids[:, start : start + 1],
                stop_token_ids=stop_token_ids,
            )
            print(f"accepted_draft_tokens={verification.accepted_draft_tokens}")
            print(f"effective_proposal_length={verification.effective_proposal_length}")
            print_tensor("verification.target_probs", verification.target_probs)
            print_tensor("verification.committed_tokens", verification.committed_tokens)
            print(f"committed_text={tokenizer.decode(verification.committed_tokens[0].tolist())!r}")

            accepted_draft_tokens = int(verification.accepted_draft_tokens)
            output_ids[:, start : start + accepted_draft_tokens + 1] = proposal.verify_input_ids[
                :, : accepted_draft_tokens + 1
            ]
            if verification.terminated_by_stop_token:
                start += accepted_draft_tokens
                past_key_values_target.crop(start)
                break
            output_ids[:, start + accepted_draft_tokens + 1] = verification.next_token
            new_token_ids = output_ids[:, start + 1 : start + accepted_draft_tokens + 2]
            start += accepted_draft_tokens + 1
            accepted_total += accepted_draft_tokens + 1
            past_key_values_target.crop(start)

            verified_target_hidden = extract_context_feature(
                verification.target_output.hidden_states,
                draft_model.target_layer_ids,
            )
            print_tensor("verified_target_hidden", verified_target_hidden)
            context.target_hidden_states = verified_target_hidden[:, : accepted_draft_tokens + 1, :]
            print_tensor("context.target_hidden_states after update", context.target_hidden_states)
            if has_stop_token(new_token_ids, stop_token_ids):
                print("Stop token generated; ending trace.")
                break

    final_ids = output_ids[:, : min(start + 1, max_length)]
    print("\n=== Final partial output ===")
    print(f"accepted_total_in_trace={accepted_total}")
    print(tokenizer.decode(final_ids[0].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
