"""真实TokenizePrompt函数体的Guidance条件回归（无需下载模型）。"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1]


def get_tokenize_prompt():
    path = ROOT / "src/openpi/transforms.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenizePrompt")
    mod = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    namespace = {
        "dataclasses": dataclasses, "DataTransformFn": object,
        "_tokenizer": NS(PaligemmaTokenizer=object),
    }
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), namespace)
    return namespace["TokenizePrompt"]


class StubTokenizer:
    def tokenize(self, prompt, state, advantage, adv_ind_dropout=True):
        # 标记原始 Advantage 输入，以验证实际是否创建了两套不同Token。
        return [advantage if advantage is not None else "unconditional"], [True]


class GuidanceTokenTests(unittest.TestCase):
    def test_recap_builds_positive_and_unconditional_tokens(self):
        Tokenize = get_tokenize_prompt()
        transform = Tokenize(tokenizer=StubTokenizer(), adv_ind_input=True,
                             adv_ind_dropout=False, adv_guidance_input=True)
        result = transform({"prompt": "pick cube", "adv_ind": "positive"})
        self.assertEqual(result["tokenized_prompt"], ["positive"])
        self.assertEqual(result["tokenized_prompt_uncond"], ["unconditional"])
        self.assertIn("tokenized_prompt_uncond_mask", result)

    def test_beta1_has_no_unconditional_tokens(self):
        Tokenize = get_tokenize_prompt()
        transform = Tokenize(tokenizer=StubTokenizer(), adv_ind_input=True,
                             adv_ind_dropout=False, adv_guidance_input=False)
        result = transform({"prompt": "pick cube", "adv_ind": "positive"})
        self.assertEqual(result["tokenized_prompt"], ["positive"])
        self.assertNotIn("tokenized_prompt_uncond", result)

    def test_mixed_sft_drops_advantage_even_when_data_contains_labels(self):
        Tokenize = get_tokenize_prompt()
        transform = Tokenize(tokenizer=StubTokenizer(), adv_ind_input=False,
                             adv_ind_dropout=False, adv_guidance_input=False)
        result = transform({"prompt": "pick cube", "adv_ind": "negative"})
        self.assertEqual(result["tokenized_prompt"], ["unconditional"])
        self.assertNotIn("tokenized_prompt_uncond", result)

    def test_libero_transform_factory_wires_guidance_field(self):
        tree = ast.parse((ROOT / "src/openpi/training/config.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelTransformFactory")
        keywords = [
            keyword.arg
            for n in ast.walk(cls) if isinstance(n, ast.Call)
            for keyword in n.keywords
        ]
        self.assertIn("adv_guidance_input", keywords)


if __name__ == "__main__":
    unittest.main()
