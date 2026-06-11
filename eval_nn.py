"""Evaluate NN crackers on the fixed benchmark set.

Usage: python eval_nn.py v3|v4|v4_greedy
"""
import json
import sys
import io
import torch

if not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from benchmark import accuracy

SEQ_LEN = 50
D_MODEL = 256
NHEAD = 4
LAYERS = 4


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "v4"
    with open("benchmark_set.json", "r", encoding="utf-8") as f:
        cases = json.load(f)

    if which == "v3":
        from nn_model_v3 import MetaTransformerCracker
        from nn_model import JamoTokenizer

        jamo_chars = [chr(i) for i in range(0x3131, 0x3164)] + [" "]
        tokenizer = JamoTokenizer(jamo_chars)
        model = MetaTransformerCracker(tokenizer.vocab_size, D_MODEL, NHEAD, LAYERS)
        model.load_state_dict(torch.load("jamo_nn_v3_fair.pth", map_location="cpu"))
        model.eval()

        @torch.no_grad()
        def solve(ct):
            src = torch.tensor(tokenizer.encode(ct)).unsqueeze(1)
            out = model(src)
            pred = out.argmax(dim=2).squeeze(1).tolist()
            return tokenizer.decode(pred)

    elif which.startswith("v6"):
        from nn_v6 import DualRankTransformerCracker, decode_constrained, decode_greedy

        model = DualRankTransformerCracker(d_model=256, nhead=8, num_layers=6,
                                           max_len=210, norm_first=True)
        pth = "jamo_nn_v6.pth"
        model.load_state_dict(torch.load(pth, map_location="cpu"))
        model.eval()
        if which.endswith("greedy"):
            solve = lambda ct: decode_greedy(model, ct)
        else:
            solve = lambda ct: decode_constrained(model, ct)

    elif which.startswith("v5"):
        from nn_v5 import RankTransformerCracker, decode_constrained, decode_greedy

        model = RankTransformerCracker(D_MODEL, NHEAD, LAYERS, norm_first=True)
        model.load_state_dict(torch.load("jamo_nn_v5.pth", map_location="cpu"))
        model.eval()
        if which.endswith("greedy"):
            solve = lambda ct: decode_greedy(model, ct)
        else:
            solve = lambda ct: decode_constrained(model, ct)

    else:
        from nn_v4 import RankTransformerCracker, decode_constrained, decode_greedy

        if which.startswith("v4b"):
            model = RankTransformerCracker(D_MODEL, NHEAD, LAYERS, norm_first=True)
            model.load_state_dict(torch.load("jamo_nn_v4b.pth", map_location="cpu"))
        else:
            model = RankTransformerCracker(D_MODEL, NHEAD, LAYERS)
            model.load_state_dict(
                torch.load("jamo_nn_v4_fair.pth", map_location="cpu")
            )
        model.eval()
        if which.endswith("greedy"):
            solve = lambda ct: decode_greedy(model, ct)
        else:
            solve = lambda ct: decode_constrained(model, ct)

    accs = []
    for i, case in enumerate(cases):
        dec = solve(case["ciphertext"])
        acc = accuracy(case["plaintext"], dec)
        accs.append(acc)
        print(f"[{i+1}/{len(cases)}] acc={acc:.2%}")
        print(f"  orig: {case['plaintext']}")
        print(f"  dec:  {dec}")

    avg = sum(accs) / len(accs)
    print(f"\n=== NN {which} ===")
    print(f"Average accuracy: {avg:.2%}")
    print(f"Cases >=90%: {sum(1 for a in accs if a >= 0.9)}/{len(accs)}")


if __name__ == "__main__":
    main()
