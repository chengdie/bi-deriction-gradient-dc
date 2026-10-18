"""Reproducible inputs and append-only analysis artifacts (stdlib only)."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone


SCHEMA_VERSION = "online-proxy-fd-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL: {path}:{line_number}") from exc


def inspect_inputs(files):
    info = []
    for file in files:
        path = Path(file).resolve()
        with path.open("rb") as stream:
            header = stream.readline()
        if header.rstrip() == b"version https://git-lfs.github.com/spec/v1":
            raise ValueError(f"Git LFS pointer, not training data: {path}. Fetch the actual file from "
                             "https://huggingface.co/datasets/princeton-nlp/less_data and verify its LFS SHA256.")
        count = 0
        for row in read_jsonl(path):
            if not isinstance(row, dict) or not (row.get("messages") or
                                                ("prompt" in row and "completion" in row)):
                raise ValueError(f"Expected raw instruction messages in {path}, row {count}")
            count += 1
        if not count:
            raise ValueError(f"Empty training input: {path}")
        info.append(dict(path=str(path), rows=count, sha256=sha256_file(path)))
    if len({item["path"] for item in info}) != len(info):
        raise ValueError("Duplicate training files")
    return info


class JsonlPool:
    """Keep byte offsets, not the entire instruction corpus, in memory."""
    def __init__(self, files):
        self.files = [Path(path).resolve() for path in files]
        self.offsets = []
        for source, path in enumerate(self.files):
            with path.open("rb") as stream:
                row = 0
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append((source, row, offset))
                    row += 1

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        source, row, offset = self.offsets[index]
        with self.files[source].open("rb") as stream:
            stream.seek(offset)
            record = json.loads(stream.readline())
        identity = dict(pool_index=index, id=str(record.get("id", f"{self.files[source].parent.name}:{row}")),
                        source=str(self.files[source]), source_row=row)
        return record, identity


def encode_reference_records(task, data_dir, tokenizer, max_length):
    from less.data_selection.target_data import load_reference_records
    from less.data_selection.get_training_dataset import encode_with_qwen_chat_format
    raw = load_reference_records(task, data_dir)
    result = []
    for record in raw:
        text = tokenizer.apply_chat_template(record["messages"], tokenize=False,
                                            add_generation_prompt=False, enable_thinking=False)
        if len(tokenizer.encode(text, add_special_tokens=False)) > max_length:
            raise ValueError(f"Reference {record['id']} exceeds reference-max-length; do not truncate its answer")
        encoded = encode_with_qwen_chat_format(record, tokenizer, max_length)
        if not any(label != -100 for label in encoded["labels"][1:]):
            raise ValueError(f"Reference {record['id']} has no supervised answer")
        result.append(dict(encoded, id=record["id"], subtask=record["subtask"]))
    return result


def code_provenance(root):
    root = Path(root)
    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    paths = sorted(set(root.glob("less/train/online_proxy*.py")) |
                   set(root.glob("less/analysis/*fd*.py")) |
                   set(root.glob("evaluation/eval/*protocol*.py")) |
                   {root / "evaluation/eval/qwen.py", root / "less/model_utils.py",
                    root / "less/data_selection/target_data.py", root / "less/data_selection/get_training_dataset.py"})
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                working_tree=git("status", "--porcelain"),
                source_sha256={str(path.relative_to(root)): sha256_file(path) for path in paths if path.is_file()})


def environment_info(torch, device):
    gpu = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = dict(name=props.name, total_memory_gb=props.total_memory / 1e9,
                   capability=list(torch.cuda.get_device_capability(device)))
    driver = None
    if device.type == "cuda":
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10)
            driver = sorted(set(result.stdout.splitlines())) if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            pass
    versions = {}
    from importlib.metadata import version, PackageNotFoundError
    for package in ("transformers", "peft", "datasets", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions,
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=gpu, driver=driver)


class RunWriter:
    def __init__(self, output):
        self.output = Path(output)
        if self.output.exists() and (not self.output.is_dir() or any(self.output.iterdir())):
            raise FileExistsError(f"Run output must be empty: {self.output}")
        self.output.mkdir(parents=True, exist_ok=True)
        self.streams = {name: (self.output / f"{name}.jsonl").open("x", encoding="utf-8")
                        for name in ("batches", "decisions", "accepted_ids", "refreshes", "checkpoints")}

    def append(self, name, value):
        stream = self.streams[name]
        stream.write(json.dumps(dict(schema_version=SCHEMA_VERSION, **value), ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()

    def batch(self, row):
        self.append("batches", row)
        for attempt in row["attempts"]:
            for decision in attempt["decisions"]:
                self.append("decisions", dict(batch_index=row["batch_index"], epoch=row["epoch"],
                                              update_index=row["update_index"], attempt=attempt["attempt"],
                                              final_attempt=attempt is row["attempts"][-1], **decision))
        for decision in row["decisions"]:
            if decision["accepted"]:
                self.append("accepted_ids", dict(batch_index=row["batch_index"], epoch=row["epoch"], **decision))
        for event in row["refreshes"]:
            self.append("refreshes", event)

    def close(self):
        for stream in self.streams.values():
            stream.close()
