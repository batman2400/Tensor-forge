"""Fine-tune a multilingual encoder with category, secondary, and urgent heads.

Device-agnostic: the laptop can smoke-test it, and the same command runs 5-fold
CV on a Kaggle T4. Stage A (train to validation) is the number to quote. The
pooled out-of-fold score is only a ranking check, same caveat as the classical
models, because near-duplicate templates cross fold boundaries.

    python -m ml.train_encoder --data-dir data --out-dir ml/runs/encoder_e5_small

Checkpoints are the best epoch per split (macro-F1 on that split's held-out
rows). A finished split is skipped on the next launch, so a dead Kaggle session
can be re-run without repeating completed folds.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.inference import apply_rules
from app.labels import CATEGORIES, SECONDARY_CATEGORIES
from app.text import build_input
from ml.data import TicketRow, assign_folds, load_all, load_folds
from ml.evaluate import EXPERIMENTS_PATH, append_experiment, compute_metrics
from ml.truncation import HEAD_TOKENS, MAX_LENGTH, TAIL_TOKENS, assemble_ids

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
THRESHOLDS = {"secondary": 0.5, "urgent": 0.5, "review": 0.5}
LOSS_SECONDARY = 0.5
LOSS_URGENT = 0.5


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    epochs: int
    batch_size: int
    lr: float
    patience: int
    seed: int
    smoke: bool


class MultiHeadEncoder(nn.Module):
    """Shared encoder, mean-pooled, then three task heads."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.category = nn.Linear(hidden, len(CATEGORIES))
        self.secondary = nn.Linear(hidden, len(SECONDARY_CATEGORIES))
        self.urgent = nn.Linear(hidden, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = self.dropout(pooled)
        urgent = self.urgent(pooled).squeeze(-1)
        return self.category(pooled), self.secondary(pooled), urgent


class EncodedTickets(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category: torch.Tensor,
        secondary: torch.Tensor,
        urgent: torch.Tensor,
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.category = category
        self.secondary = secondary
        self.urgent = urgent

    def __len__(self) -> int:
        return int(self.category.shape[0])

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_ids[index],
            self.attention_mask[index],
            self.category[index],
            self.secondary[index],
            self.urgent[index],
        )


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _encode_texts(tokenizer: Any, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise RuntimeError("tokenizer has no CLS/SEP ids; refusing a silent truncation change")
    input_ids = torch.full((len(texts), MAX_LENGTH), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(texts), MAX_LENGTH), dtype=torch.long)
    for row, text in enumerate(texts):
        raw = tokenizer.encode(text, add_special_tokens=False)
        ids = assemble_ids(list(raw), int(cls_id), int(sep_id))
        width = len(ids)
        input_ids[row, :width] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :width] = 1
    return input_ids, attention_mask


def _label_tensors(rows: list[TicketRow]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    category_index = {label: index for index, label in enumerate(CATEGORIES)}
    secondary_index = {label: index for index, label in enumerate(SECONDARY_CATEGORIES)}
    category = torch.tensor([category_index[row.category] for row in rows], dtype=torch.long)
    secondary = torch.zeros((len(rows), len(SECONDARY_CATEGORIES)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        label = row.secondary_category
        if label in secondary_index:
            secondary[row_index, secondary_index[label]] = 1.0
    urgent = torch.tensor([1.0 if row.is_urgent else 0.0 for row in rows], dtype=torch.float32)
    return category, secondary, urgent


def _class_weights(category: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(category, minlength=len(CATEGORIES)).float()
    weights = counts.sum() / counts.clamp(min=1.0)
    return weights / weights.mean()


def _pos_weight(binary: torch.Tensor) -> torch.Tensor:
    if binary.ndim == 1:
        positive = binary.sum()
        negative = binary.numel() - positive
        return (negative / positive.clamp(min=1.0)).reshape(1)
    positive = binary.sum(dim=0)
    negative = binary.shape[0] - positive
    return negative / positive.clamp(min=1.0)


def _texts(rows: list[TicketRow]) -> list[str]:
    return [build_input(row.channel, row.subject, row.text) for row in rows]


def _predictions(
    category_proba: np.ndarray,
    secondary_proba: np.ndarray,
    urgent_proba: np.ndarray,
) -> list[dict[str, Any]]:
    labels = list(CATEGORIES)
    secondary_labels = list(SECONDARY_CATEGORIES)
    predictions: list[dict[str, Any]] = []
    for index in range(category_proba.shape[0]):
        predictions.append(
            apply_rules(
                category_proba[index],
                labels,
                secondary_proba[index],
                secondary_labels,
                float(urgent_proba[index]),
                THRESHOLDS,
            )
        )
    return predictions


def _score(
    rows: list[TicketRow],
    predictions: list[dict[str, Any]],
    category_proba: np.ndarray,
    urgent_proba: np.ndarray,
) -> dict[str, Any]:
    return compute_metrics(
        gold_category=[row.category for row in rows],
        pred_category=[item["category"] for item in predictions],
        category_proba=category_proba,
        gold_secondary=[row.secondary_category for row in rows],
        pred_secondary=[item["secondary_category"] for item in predictions],
        gold_urgent=[row.is_urgent for row in rows],
        pred_urgent=[bool(item["is_urgent"]) for item in predictions],
        urgent_proba=urgent_proba,
        languages=[row.language for row in rows],
        channels=[row.channel for row in rows],
    )


def _jsonable(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop nothing; metrics are already plain numbers, lists, and dicts."""
    return metrics


@torch.no_grad()
def _predict(
    model: MultiHeadEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    use_amp = device.type == "cuda"
    category_parts: list[np.ndarray] = []
    secondary_parts: list[np.ndarray] = []
    urgent_parts: list[np.ndarray] = []
    for start in range(0, input_ids.shape[0], batch_size):
        stop = start + batch_size
        ids = input_ids[start:stop].to(device, non_blocking=True)
        mask = attention_mask[start:stop].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            category_logits, secondary_logits, urgent_logits = model(ids, mask)
        category_parts.append(torch.softmax(category_logits.float(), dim=-1).cpu().numpy())
        secondary_parts.append(torch.sigmoid(secondary_logits.float()).cpu().numpy())
        urgent_parts.append(torch.sigmoid(urgent_logits.float()).cpu().numpy())
    return (
        np.concatenate(category_parts, axis=0),
        np.concatenate(secondary_parts, axis=0),
        np.concatenate(urgent_parts, axis=0),
    )


def _train_one(
    train_rows: list[TicketRow],
    eval_rows: list[TicketRow],
    tokenizer: Any,
    encoder: nn.Module,
    config: RunConfig,
    directory: Path,
    device: torch.device,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    proba_path = directory / "proba.npz"
    if metrics_path.is_file() and proba_path.is_file():
        print(f"skip {directory.name}: checkpoint already finished", flush=True)
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        return saved

    _seed_everything(config.seed)
    train_ids, train_mask = _encode_texts(tokenizer, _texts(train_rows))
    eval_ids, eval_mask = _encode_texts(tokenizer, _texts(eval_rows))
    train_cat, train_sec, train_urg = _label_tensors(train_rows)
    dataset = EncodedTickets(train_ids, train_mask, train_cat, train_sec, train_urg)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = MultiHeadEncoder(encoder).to(device)
    category_weight = _class_weights(train_cat).to(device)
    secondary_pos = _pos_weight(train_sec).to(device)
    urgent_pos = _pos_weight(train_urg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for ids, mask, cat_y, sec_y, urg_y in loader:
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            cat_y = cat_y.to(device, non_blocking=True)
            sec_y = sec_y.to(device, non_blocking=True)
            urg_y = urg_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                cat_logits, sec_logits, urg_logits = model(ids, mask)
            loss_cat = nn.functional.cross_entropy(
                cat_logits.float(), cat_y, weight=category_weight
            )
            loss_sec = nn.functional.binary_cross_entropy_with_logits(
                sec_logits.float(), sec_y, pos_weight=secondary_pos
            )
            loss_urg = nn.functional.binary_cross_entropy_with_logits(
                urg_logits.float().unsqueeze(1),
                urg_y.unsqueeze(1),
                pos_weight=urgent_pos,
            )
            loss = loss_cat + LOSS_SECONDARY * loss_sec + LOSS_URGENT * loss_urg
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * ids.shape[0]
            seen += int(ids.shape[0])

        category_proba, secondary_proba, urgent_proba = _predict(
            model, eval_ids, eval_mask, config.batch_size * 2, device
        )
        metrics = _score(
            eval_rows,
            _predictions(category_proba, secondary_proba, urgent_proba),
            category_proba,
            urgent_proba,
        )
        print(
            f"{directory.name} epoch {epoch} loss {running / max(seen, 1):.4f} "
            f"macro_f1 {metrics['macro_f1']:.4f} acc {metrics['accuracy']:.4f} "
            f"urgent_f1 {metrics['urgent_f1']:.4f}",
            flush=True,
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = float(metrics["macro_f1"])
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= config.patience:
                print(f"{directory.name} early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"no epoch completed for {directory}")
    model.load_state_dict(best_state)
    category_proba, secondary_proba, urgent_proba = _predict(
        model, eval_ids, eval_mask, config.batch_size * 2, device
    )
    predictions = _predictions(category_proba, secondary_proba, urgent_proba)
    metrics = _score(eval_rows, predictions, category_proba, urgent_proba)
    metrics["best_epoch"] = best_epoch
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.save(best_state, directory / "model.pt")
    np.savez_compressed(
        proba_path,
        ticket_id=np.array([row.ticket_id for row in eval_rows]),
        category_proba=category_proba,
        secondary_proba=secondary_proba,
        urgent_proba=urgent_proba,
    )
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2) + "\n", encoding="utf-8")
    return metrics


def _train_final(
    rows: list[TicketRow],
    tokenizer: Any,
    encoder: nn.Module,
    config: RunConfig,
    directory: Path,
    device: torch.device,
    accum_steps: int,
) -> None:
    """Train on every row for a fixed epoch count. No held-out early stop.

    The epoch count is the stage-A best epoch. Gradient accumulation keeps the
    effective batch at ``batch_size * accum_steps`` on a 6 GB GPU.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / "model.pt"
    if final_path.is_file():
        print(f"skip final: {final_path} already exists", flush=True)
        return
    if accum_steps < 1:
        raise RuntimeError("accum steps must be at least 1")

    _seed_everything(config.seed)
    train_ids, train_mask = _encode_texts(tokenizer, _texts(rows))
    train_cat, train_sec, train_urg = _label_tensors(rows)
    dataset = EncodedTickets(train_ids, train_mask, train_cat, train_sec, train_urg)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = MultiHeadEncoder(encoder).to(device)
    category_weight = _class_weights(train_cat).to(device)
    secondary_pos = _pos_weight(train_sec).to(device)
    urgent_pos = _pos_weight(train_urg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps = max(accum_steps, 1)

    for epoch in range(1, config.epochs + 1):
        epoch_path = directory / f"epoch_{epoch}.pt"
        if epoch_path.is_file():
            print(f"final epoch {epoch} already saved", flush=True)
            continue
        previous = directory / f"epoch_{epoch - 1}.pt"
        if previous.is_file():
            model.load_state_dict(torch.load(previous, map_location=device, weights_only=True))
        model.train()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (ids, mask, cat_y, sec_y, urg_y) in enumerate(loader, start=1):
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            cat_y = cat_y.to(device, non_blocking=True)
            sec_y = sec_y.to(device, non_blocking=True)
            urg_y = urg_y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                cat_logits, sec_logits, urg_logits = model(ids, mask)
            loss_cat = nn.functional.cross_entropy(
                cat_logits.float(), cat_y, weight=category_weight
            )
            loss_sec = nn.functional.binary_cross_entropy_with_logits(
                sec_logits.float(), sec_y, pos_weight=secondary_pos
            )
            loss_urg = nn.functional.binary_cross_entropy_with_logits(
                urg_logits.float().unsqueeze(1),
                urg_y.unsqueeze(1),
                pos_weight=urgent_pos,
            )
            loss = loss_cat + LOSS_SECONDARY * loss_sec + LOSS_URGENT * loss_urg
            scaler.scale(loss / steps).backward()
            if step % steps == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.item()) * ids.shape[0]
            seen += int(ids.shape[0])
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(state, epoch_path)
        print(
            f"final epoch {epoch} loss {running / max(seen, 1):.4f} saved {epoch_path.name}",
            flush=True,
        )

    last = directory / f"epoch_{config.epochs}.pt"
    if not last.is_file():
        raise RuntimeError(f"final training stopped before epoch {config.epochs}")
    last.replace(final_path)
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"wrote {final_path}", flush=True)


def _load_encoder(model_name: str) -> tuple[Any, nn.Module]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    return tokenizer, encoder


def _fresh_encoder(model_name: str) -> nn.Module:
    from transformers import AutoModel

    return AutoModel.from_pretrained(model_name)


def _subsample(rows: list[TicketRow], limit: int, seed: int) -> list[TicketRow]:
    if len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    picked = rng.choice(len(rows), size=limit, replace=False)
    return [rows[int(index)] for index in sorted(picked)]


def _headline(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "urgent_f1": float(metrics["urgent_f1"]),
        "secondary_exact": float(metrics["secondary_exact"]),
    }


def _pool_fold_metrics(rows: list[TicketRow], out_dir: Path, n_splits: int) -> dict[str, Any]:
    by_id = {row.ticket_id: row for row in rows}
    order = [row.ticket_id for row in rows]
    category = np.zeros((len(rows), len(CATEGORIES)), dtype=np.float64)
    secondary = np.zeros((len(rows), len(SECONDARY_CATEGORIES)), dtype=np.float64)
    urgent = np.zeros(len(rows), dtype=np.float64)
    filled = np.zeros(len(rows), dtype=bool)
    position = {ticket_id: index for index, ticket_id in enumerate(order)}
    for fold in range(n_splits):
        payload = np.load(out_dir / f"fold_{fold}" / "proba.npz", allow_pickle=False)
        for row_index, ticket_id in enumerate(payload["ticket_id"].tolist()):
            slot = position[str(ticket_id)]
            category[slot] = payload["category_proba"][row_index]
            secondary[slot] = payload["secondary_proba"][row_index]
            urgent[slot] = payload["urgent_proba"][row_index]
            filled[slot] = True
    if not bool(filled.all()):
        missing = int((~filled).sum())
        raise RuntimeError(f"{missing} tickets have no out-of-fold prediction")
    heldout = [by_id[ticket_id] for ticket_id in order]
    predictions = _predictions(category, secondary, urgent)
    return _score(heldout, predictions, category, urgent)


def run(
    data_dir: Path, out_dir: Path, config: RunConfig, skip_stage_a: bool, skip_cv: bool
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"device {device.type} model {config.model_name}", flush=True)
    log_experiment = not (out_dir / "experiment.json").is_file()
    rows = load_all(data_dir)
    if config.smoke:
        train_rows = _subsample([row for row in rows if row.split == "train"], 128, config.seed)
        eval_rows = _subsample([row for row in rows if row.split == "validation"], 32, config.seed)
        rows = train_rows + eval_rows
    train_rows = [row for row in rows if row.split == "train"]
    valid_rows = [row for row in rows if row.split == "validation"]

    tokenizer, template = _load_encoder(config.model_name)
    summary: dict[str, Any] = {
        "model_name": config.model_name,
        "max_length": MAX_LENGTH,
        "head_tokens": HEAD_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "thresholds": THRESHOLDS,
        "smoke": config.smoke,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_name": config.model_name,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "patience": config.patience,
                "seed": config.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stage_a: dict[str, Any] | None = None
    if not skip_stage_a:
        # Each split needs its own weight init. Reuse the already-downloaded object for stage A.
        stage_a = _train_one(
            train_rows,
            valid_rows,
            tokenizer,
            template,
            config,
            out_dir / "stage_a",
            device,
        )
        summary["stage_a"] = _headline(stage_a)
        print(
            "stage A "
            f"macro_f1 {summary['stage_a']['macro_f1']:.4f} "
            f"acc {summary['stage_a']['accuracy']:.4f} "
            f"urgent_f1 {summary['stage_a']['urgent_f1']:.4f}",
            flush=True,
        )
    del template

    cv: dict[str, Any] | None = None
    if not skip_cv and not config.smoke:
        folds_path = Path(__file__).resolve().parent / "folds.json"
        fold_of = load_folds(folds_path) if folds_path.is_file() else assign_folds(rows)
        n_splits = 5
        for fold in range(n_splits):
            held_in = [row for row in rows if fold_of[row.ticket_id] != fold]
            held_out = [row for row in rows if fold_of[row.ticket_id] == fold]
            _train_one(
                held_in,
                held_out,
                tokenizer,
                _fresh_encoder(config.model_name),
                RunConfig(
                    model_name=config.model_name,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    patience=config.patience,
                    seed=config.seed + fold,
                    smoke=False,
                ),
                out_dir / f"fold_{fold}",
                device,
            )
        cv = _pool_fold_metrics(rows, out_dir, n_splits)
        summary["cv"] = _headline(cv)
        (out_dir / "cv_metrics.json").write_text(json.dumps(cv, indent=2) + "\n", encoding="utf-8")
        print(
            "cv "
            f"macro_f1 {summary['cv']['macro_f1']:.4f} "
            f"acc {summary['cv']['accuracy']:.4f} "
            f"urgent_f1 {summary['cv']['urgent_f1']:.4f}",
            flush=True,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if log_experiment and not config.smoke and (stage_a is not None or cv is not None):
        row = {
            "id": "encoder_e5_small",
            "branch": "encoder",
            "features": "multilingual-e5-small mean-pool head+tail 256",
            "hyperparameters": (
                f"AdamW lr={config.lr} epochs<={config.epochs} patience={config.patience} "
                f"batch={config.batch_size} class_weight+pos_weight fp16 thresholds 0.5"
            ),
            "notes": "Stage A is the number to quote. CV/OOF is a ranking check only.",
        }
        if stage_a is not None:
            row["stage_a_accuracy"] = stage_a["accuracy"]
            row["stage_a_macro_f1"] = stage_a["macro_f1"]
            row["stage_a_urgent_f1"] = stage_a["urgent_f1"]
            row["stage_a_secondary_exact"] = stage_a["secondary_exact"]
        if cv is not None:
            row["cv_accuracy"] = cv["accuracy"]
            row["cv_macro_f1"] = cv["macro_f1"]
            row["cv_urgent_f1"] = cv["urgent_f1"]
            row["cv_secondary_exact"] = cv["secondary_exact"]
        if EXPERIMENTS_PATH.parent.is_dir():
            append_experiment(row)
        (out_dir / "experiment.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multilingual encoder heads.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("ml/runs/encoder_e5_small"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="One short stage-A run on a few rows.")
    parser.add_argument("--skip-stage-a", action="store_true")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument(
        "--final",
        action="store_true",
        help="Train on all tickets for a fixed epoch count and save model.pt.",
    )
    parser.add_argument("--accum-steps", type=int, default=1)
    args = parser.parse_args()
    if args.final:
        out_dir = args.out_dir
        if out_dir == Path("ml/runs/encoder_e5_small"):
            out_dir = Path("ml/runs/encoder_e5_final")
        rows = load_all(args.data_dir)
        tokenizer, encoder = _load_encoder(args.model)
        config = RunConfig(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            seed=args.seed,
            smoke=False,
        )
        _train_final(
            rows,
            tokenizer,
            encoder,
            config,
            out_dir / "final",
            _device(),
            args.accum_steps,
        )
        return
    out_dir = args.out_dir
    if args.smoke and out_dir == Path("ml/runs/encoder_e5_small"):
        out_dir = Path("ml/runs/encoder_smoke")
    epochs = 1 if args.smoke else args.epochs
    batch_size = 8 if args.smoke else args.batch_size
    config = RunConfig(
        model_name=args.model,
        epochs=epochs,
        batch_size=batch_size,
        lr=args.lr,
        patience=1 if args.smoke else args.patience,
        seed=args.seed,
        smoke=args.smoke,
    )
    run(
        args.data_dir,
        out_dir,
        config,
        skip_stage_a=args.skip_stage_a,
        skip_cv=True if args.smoke else args.skip_cv,
    )


if __name__ == "__main__":
    main()
