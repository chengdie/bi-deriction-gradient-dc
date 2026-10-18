"""Auditable LESS benchmark settings, separate from the Qwen-native protocol.

Published runs did not export sample manifests. Their filesystem/set iteration
order is not reconstructible from a seed alone. This profile freezes a sorted,
deterministic selection and explicitly records that limitation. Supply the SAME
exported eval manifest to every model being directly compared.
"""

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import string

from less.data_selection.target_data import _raw


def load_eval_data(task, data_dir, seed=42, max_per_subtask=None):
    refs, tests = _raw(task, data_dir)
    demonstrations, groups = defaultdict(list), defaultdict(list)
    for row in refs:
        demonstrations[row["subtask"]].append(row)
    for row in tests:
        groups[row["subtask"]].append(row)
    cap = max_per_subtask if max_per_subtask is not None else {"mmlu": None, "bbh": 40, "tydiqa": 200}[task]
    rng, records = random.Random(seed), []
    for subject, rows in sorted(groups.items()):
        if cap and len(rows) > cap:
            rows = rng.sample(rows, cap)
        if subject not in demonstrations:
            raise ValueError(f"Missing demonstrations for {subject}")
        for row in rows:
            record = dict(id=row["id"], subtask=subject, answers=row["answers"], less_task=task,
                          question=row["question"], demonstrations=demonstrations[subject],
                          fewshot_ids=[demo["id"] for demo in demonstrations[subject]])
            if task == "mmlu":
                record["choices"] = list("ABCD")
            elif task == "bbh":
                prompt_file = Path(data_dir) / "eval" / "bbh" / "cot-prompts" / f"{subject}.txt"
                record["less_prompt"] = "".join(prompt_file.read_text().splitlines(keepends=True)[2:]).strip() + "\n\nQ: " + row["question"] + "\nA:"
            else:
                record["context"] = row["context"]
            records.append(record)
    digest = hashlib.sha256(json.dumps(records, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    reference_questions = {" ".join(row["question"].casefold().split()) for row in refs}
    overlap_ids = [record["id"] for record in records if " ".join(record["question"].casefold().split()) in reference_questions]
    return records, dict(version="less-benchmark-v1", task=task, seed=seed,
        shots={"mmlu": 5, "bbh": 3, "tydiqa": 1}[task], max_per_subtask=cap,
        record_sha256=digest, test_records=len(records),
        aggregation="question micro mean (legacy average_acc)" if task == "mmlu" else "subtask macro mean",
        sampling="sorted subtasks, shared Python Random(seed); frozen in eval_manifest.json",
        historical_sample_identity="not claimed without the original run's sample manifest",
        demonstrations="provided dev/one-shot records; TyDiQA train-sampling was not regenerated",
        chat_format="raw benchmark prompts; Tulu wrapper for TyDiQA", enable_thinking=None,
        subtasks=sorted(groups), n_subtasks=len(groups), expected_subtasks={"mmlu": 57, "bbh": 27, "tydiqa": 9}[task],
        reference_overlap_ids=overlap_ids, reference_overlap_n=len(overlap_ids),
        context_policy="MMLU reduce shots above 2048 tokens; TyDiQA passage truncated to 512 tokenizer tokens",
        reference_overlap_policy="unaltered legacy test split; reference data loaded separately by training")


def _tydi_templates():
    # Read the original constant without importing its optional vllm/evaluate deps.
    path = Path(__file__).parent / "tydiqa" / "run_eval.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
                target.id == "encoding_templates_with_context" for target in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Original TyDiQA templates are missing")


def prompt(record, tokenizer, max_length, shots=None):
    task = record["less_task"]
    if task == "bbh":
        return record["less_prompt"], 3
    if task == "mmlu":
        count = 5 if shots is None else shots
        while count >= 0:
            header = "The following are multiple choice questions (with answers) about " + " " + record["subtask"].replace("_", " ") + ".\n\n"
            demonstrations = "".join(row["question"] + "\nAnswer: " + row["completion"] + "\n\n"
                                     for row in record["demonstrations"][:count])
            text = header + demonstrations + record["question"] + "\nAnswer:"
            if len(tokenizer.encode(text, add_special_tokens=False)) <= min(2048, max_length):
                return text, count
            count -= 1
        raise ValueError(f"MMLU question exceeds context even without demonstrations: {record['id']}")
    language = record["subtask"]
    instruction, passage_label, question_label, answer_label = _tydi_templates()[language]
    def passage(value):
        tokens = tokenizer.encode(value)
        return tokenizer.decode(tokens[:512]) if len(tokens) > 512 else value
    demonstration = record["demonstrations"][0]
    text = (instruction + "\n\n" + passage_label + " " + passage(demonstration["context"]) + "\n" +
            question_label + " " + demonstration["question"] + "\n" + answer_label + " " +
            demonstration["completion"] + "\n\n" + passage_label + " " + passage(record["context"]) +
            "\n" + question_label + " " + record["question"] + "\n")
    return "<|user|>\n" + text + "\n<|assistant|>\n" + answer_label, 1


def extract_answer(text, task):
    if task == "bbh":
        found = re.search(r"So the answer is (.*?)\.", text)
        return found.group(1).strip() if found else text.strip().split("\n\n")[0].strip()
    return text.strip().split("\n")[0].strip()


def score(task, prediction, answers):
    if task != "bbh":
        from evaluation.eval.qwen import score as native_score
        return native_score(task, prediction, answers)
    normalize = lambda value: value.lower().translate(str.maketrans("", "", string.punctuation))
    return {"exact_match": max(float(normalize(prediction) == normalize(answer)) for answer in answers)}
