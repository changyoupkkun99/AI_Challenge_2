"""재사용 VQA 베이스라인의 초기 설정. 이 파일은 학습 실행기가 아니다.

기본값은 A186 ZIP의 최종 Gemma 학습 코드를 참고한다. 분할 4:1,
학습 정답 위치 균등화, 실제 logits/확률 저장은 새 베이스 요구를 반영한다.
원본 노트북을 수정하지 않고 이전 모델의 설정을 별도 프로필로 보존한다.

python VQA_Baseline_Config.py --output VQA_Default_Config.json
python VQA_Baseline_Config.py --preset qwen27_only --hardware portable
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


# 새 기본값. 이전 선택지는 삭제하지 않고 아래와 PRESETS에 남긴다.
ACTIVE_PRESET = "reference_gemma31"
# ACTIVE_PRESET = "team_common"
# ACTIVE_PRESET = "qwen27_only"
# ACTIVE_PRESET = "gemma31_legacy"
# ACTIVE_PRESET = "agnes_legacy"

# ZIP 기본 배치 16 x 1, 추론 배치 32를 보존한다.
# GPU가 다르면 portable로 변경한 뒤 실제 backward/VRAM을 확인한다.
ACTIVE_HARDWARE = "reference_b200"
# ACTIVE_HARDWARE = "portable"  # batch 4 x accumulation 4, 추론 batch 4

HARDWARE_PROFILES = {
    "reference_b200": {
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "per_device_eval_batch_size": 32,
        "evidence": "A186 최종 소스 기본값 및 B200 192GB 주석; 실행 로그 미제공",
    },
    "portable": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "per_device_eval_batch_size": 4,
        "evidence": "새 베이스용 시작값; GPU 적합성/동일 성능은 실측 필요",
    },
}

SYSTEM_PROMPT = (
    "당신은 이미지를 보고 질문에 답하는 시각 질의응답 도우미입니다. "
    "a, b, c, d 중 정확히 한 글자로만 답하고, 설명은 하지 마세요."
)
USER_INSTRUCTION = "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."

BASE = {
    "schema_version": 1,
    "artifact_kind": "configuration_only",
    "run": {"seed": 42, "run_id": None, "output_root": "runs", "mode": "finetune"},
    "data": {
        "train_csv": None, "dev_csv": None, "test_csv": None,
        "image_root": None, "sample_submission_csv": None,
        "columns": {"id": "id", "image": "path", "question": "question",
                    "choices": ["a", "b", "c", "d"], "answer": "answer"},
        "classes": ["a", "b", "c", "d"],
        "label_to_index": {"a": 0, "b": 1, "c": 2, "d": 3},
        "split": {
            "train_fraction": 0.8, "validation_fraction": 0.2,
            "method": "group_stratified", "group_key": "image_sha256",
            "optional_phash_grouping": False,
            "validation_sources": ["train_gold"],
            "dev_pseudo_labels_training_only": True,
            "reuse_manifest": True, "manifest_path": None,
            "record_realized_counts_and_ratios": True,
        },
        "cleaning": {
            "mode": "manual_csv",  # precleaned_manifest / none도 별도 실험으로 선택 가능
            "manual_csv_dir": None,
            "required_files": ["train_changes.csv", "train_delete_list.csv",
                               "dev_changes.csv", "dev_delete_list.csv"],
            "apply_option_edits": True, "apply_delete_lists": True,
            "check_before_text_matches": True,
            "answer_letter_edit": False,
            "require_cleaning_provenance": True,
            "teacher_top2_review": {
                "enabled": False, "model_id": None,
                "action": "flag_for_human_review_only",
                "checkpoint_status": "글 Qwen3.8 / ZIP README Qwen3.5; 정확한 교사 미확인",
            },
        },
        "dev_votes": {
            "enabled": True, "columns_pattern": "answer[0-9]+",
            "keep_patterns": ["3", "3:1", "3:1:1"],
            "numeric_to_letter": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "target": "majority", "source_tag_required": True,
        },
        "leakage": {
            "exclude_validation_from_all_training_sources": True,
            "check_ids": True, "check_image_paths": True,
            "check_content_hash_across_sources": True,
            "check_test_overlap_and_record": True,
        },
        "training_option_order": {
            "mode": "balanced_target_positions",  # original / random_shuffle도 선택 가능
            "target_ratios": [0.25, 0.25, 0.25, 0.25],
            "deterministic_seeded": True, "remap_answer_with_options": True,
            "reject_unsafe_position_references": True,
            "report_exempt_samples_and_realized_balance": True,
            "validation_original_order": True, "test_original_order": True,
        },
        "hard_reverse": {
            "enabled": False, "scope": "training_only",
            "order": [3, 2, 1, 0], "answer_index_map": [3, 2, 1, 0],
            "evidence": "A186 실험 코드 기능; 최종 스크립트에는 없음",
        },
    },
    "image": {
        "exif_transpose": True, "color_mode": "RGB",
        "resize_mode": "native_aspect",  # letterbox 선택 시 아래 설정 사용
        "longest_side_cap": 2048, "max_soft_tokens": 1120,
        "letterbox": {"side": 1024, "fill_rgb": [255, 255, 255], "enabled": False},
        "augmentation": {
            "external": ["orig"],
            "target": ["orig", "compression", "blur", "affine"],
            "jpeg_quality": 70, "blur_radius": 1.0,
            "affine_translate_fraction": 0.03, "affine_shear_degrees": 5.0,
            "fill_rgb": [255, 255, 255],
            "validation": ["orig"], "test": ["orig"],
            "legacy_photometric": {"enabled": False,
                "source": "hybrid 노트북의 밝기/대비/색상/노이즈/JPEG 옵션 보존 대상"},
        },
    },
    "extra_context": {
        "mode": "none",  # none / ocr / dino / ocr_dino
        "ocr": {"backend": "easyocr", "languages": ["ko", "en"],
                "cache_confidence": 0.25, "prompt_confidence": 0.40, "max_items": 15},
        "dino": {"model_id": "IDEA-Research/grounding-dino-tiny",
                 "threshold": 0.30, "max_items": 12, "object_queries": None},
        "cache_key": "image_hash+preprocess_config+tool_revision",
        "preserve_question_and_image_priority": True,
        "compare_on_same_validation_manifest": True,
    },
    "prompt": {
        "system": SYSTEM_PROMPT, "instruction": USER_INSTRUCTION,
        "repeat": 2, "repeat_separator": "\n\n",
        "repeat_content": "question+four_choices+instruction",
        "image_occurrences": 1,
        "template_kwargs": {"enable_thinking": False},
        "answer_prefix": "<|channel>thought\n<channel|>",
        "append_prefix_only_if_missing": True,
        "assert_each_answer_is_one_token": True,
        "verify_next_token_answer_boundary": True,
    },
    "model": {
        "model_id": "unsloth/gemma-4-31B-it-unsloth-bnb-4bit",
        "revision": None, "record_resolved_revision": True,
        "backend": "unsloth_fastvision", "quantization": "bitsandbytes_4bit",
        "load_in_4bit": True, "compute_dtype": "bfloat16",
        "max_seq_length": 4096, "gradient_checkpointing": True,
        "trust_remote_code": False,
        "environment_before_import": {
            "UNSLOTH_DISABLE_STATISTICS": "1", "UNSLOTH_RETURN_LOGITS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    },
    "lora": {
        "r": 16, "alpha": 32, "dropout": 0.0, "bias": "none",
        "finetune_vision_layers": False, "finetune_language_layers": True,
        "finetune_attention_modules": True, "finetune_mlp_modules": True,
        "record_resolved_target_module_names": True,
    },
    "loss": {
        "mode": "next_answer_token_only", "supervised_tokens_per_sample": 1,
        "normalization": "full_vocabulary_cross_entropy",
        "exclude_prompt_image_reasoning_eos_padding_from_loss": True,
        "padding_side": "left",
    },
    "external_data": {
        "enabled": True,
        "mtvqa": {"repo_id": "ByteDance/MTVQA", "splits": ["train", "test"],
                  "languages": ["KO", "KR"], "max_answer_chars": 40,
                  "distractor_pool": 50, "convert_to_four_choices": True,
                  "distractor_priority": ["other_answers_same_image", "similar_type_and_length"],
                  "exclude_duplicate_and_gold_substring_choices": True},
        "kreta": {"repo_id": "tabtoyou/KRETA", "split": "test",
                  "choice_columns": ["A", "B", "C", "D"], "answer_to_lowercase": True},
        "dataset_revision": None, "record_resolved_revisions": True,
        "record_external_split_used_for_training": True,
    },
    "training": {
        "optimizer": "adamw_8bit", "scheduler": "cosine",
        "warmup_ratio": 0.05, "weight_decay": 0.01, "max_grad_norm": 1.0,
        "logging_steps": 10, "save_steps": 200, "save_total_limit": 2,
        "num_workers": None, "report_to": "none",
        "preflight_backward_required": True, "benchmark_steps": 20,
        "stages": [
            {"name": "external", "epochs": 1.0, "learning_rate": 1e-4,
             "augmentation": ["orig"], "save_adapter": "adapter_external"},
            {"name": "target", "epochs": 1.0, "learning_rate": 1e-4,
             "augmentation": ["orig", "compression", "blur", "affine"],
             "save_adapter": "adapter"},
        ],
        "carry_same_adapter_external_to_target": True,
        "new_optimizer_and_scheduler_per_stage": True,
        "checkpoint_selection": "last_stage_final",
        "resume_with_config_and_data_hash_checks": True,
    },
    "inference": {
        "mode": "direct_choice_logits", "temperature": 1.0,
        "choice_token_order": ["a", "b", "c", "d"],
        "shifts": [0, 2], "map_back_to_original_order": True,
        "aggregation": "arithmetic_probability_mean",
        "generation_diagnostic": {"enabled": True, "sample_count": 8},
        "validation_shift_calibration": {
            "enabled": True, "candidate_shifts": [0, 1, 2, 3],
            "apply_selected_shift_instead_of_tta": False,
            "selection_policy": "record_reference_recipe_without_automatic_search",
        },
        "highres_1024_tta4": {"enabled": False, "side": 1024,
                              "shifts": [0, 1, 2, 3], "resize_mode": "letterbox"},
        "same_model_routing": {"enabled": False, "confidence_threshold": None,
                               "margin_threshold": None},
        "save_partial_predictions": True, "resume_only_matching_run_signature": True,
    },
    "artifacts": {
        "class_order": ["a", "b", "c", "d"], "expected_counts": "derive_from_manifests",
        "required": ["probs/valid_logits.npy", "probs/valid_probs.npy",
                     "probs/valid_labels.npy", "probs/valid_ids.json",
                     "probs/test_logits.npy", "probs/test_probs.npy", "probs/test_ids.json",
                     "validation_predictions.csv", "validation_summary.json",
                     "analysis/shift_calibration.csv", "analysis/selected_shift.json"],
        "additional_tta": ["probs/valid_shift_logits.npy", "probs/valid_shift_probs.npy",
                           "probs/test_shift_logits.npy", "probs/test_shift_probs.npy",
                           "probs/valid_reference_probs.npy", "probs/test_reference_probs.npy"],
        "metadata_file": "probs/probability_metadata.json",
        "logits_semantics": "actual_restored_logits_of_reference_shift",
        "logits_reference_shift": 0,
        "probs_semantics": "final_arithmetic_mean_of_restored_shift_probabilities",
        "raw_logits_are_not_log_of_averaged_probs": True,
        "id_order": "validation_manifest_or_original_test_order",
        "submission_filename": "submission_{run_id}.csv",
        "submission_columns": ["id", "answer"],
        "charts": ["loss_lr_vram", "before_after_accuracy", "class_accuracy_confusion",
                   "shift_tta_comparison", "source_and_question_type_accuracy"],
        "run_comparison_requires_same_validation_ids_and_label_hash": True,
    },
    "environment_reference": {
        "source": "A186 ZIP requirements.txt; 실제 환경 로그/호환성 미확인",
        "python": "3.11", "cuda": "12.8", "torch": "2.11.0",
        "unsloth": "2026.9.7", "transformers": "5.5.0",
        "peft": "0.21.0", "bitsandbytes": "0.50.2",
        "record_actual_versions_and_freeze": True,
    },
    "evidence": {
        "reference_score": {"public": 0.97229,
                            "status": "solution_report_only_no_submission_or_score_log"},
        "desired_public": 0.97, "score_guaranteed": False,
        "reference_recipe_source": "A186_민구없는_민구팀.zip/src/train_gemma4_31b.py",
        "execution_status": "not_trained_not_inferred_config_prepared",
    },
}

# 현재 노트북에 있는 핵심 설정 보존. 과거 성적을 낸 설정으로 확정하지 않는다.
# 모델을 바꾸어도 새 4:1 분할·NPY 규격은 공통이다. 완전한 과거 재현에는
# 원본 노트북의 backend, manifest, processor, template, 실행 로그가 더 필요하다.
PRESETS = {
    "reference_gemma31": {"kind": "reference_recipe_for_new_baseline"},
    "team_common": {"model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                    "backend": "transformers_peft", "quantization": "none",
                    "r": 16, "alpha": 32, "lr": 2e-4, "max_seq_length": 4096,
                    "source": "팀_공통_베이스라인.ipynb"},
    "qwen27_only": {"model_id": "btbtyler09/Qwen3.8-27B-GPTQ-4bit",
                    "backend": "gptq_peft_with_backward_check", "quantization": "gptq_4bit",
                    "r": 8, "alpha": 16, "lr": 5e-5, "max_seq_length": 4096,
                    "source": "Qwen38_27B_clean7169_AgnesStyle_A100.ipynb"},
    "gemma31_legacy": {"model_id": "google/gemma-4-31B-it",
                       "backend": "transformers_peft", "quantization": "none",
                       "r": 8, "alpha": 16, "lr": 5e-5, "max_seq_length": 2048,
                       "source": "gemma4_31b_TRAINING(1).ipynb"},
    "agnes_legacy": {"model_id": "Agnes-AI/Agnes-3.0-Flash",
                     "backend": "transformers_peft_delta_targets", "quantization": "none",
                     "r": 8, "alpha": 16, "lr": 5e-5, "max_seq_length": 2048,
                     "source": "Agnes-3.0-Flash_clean7169_full_v3(2)(1).ipynb"},
}


def build_config(preset=ACTIVE_PRESET, hardware=ACTIVE_HARDWARE, extra_context="none"):
    """設定を独立して返す。モデル/データをロードしたり GPU を使用したりしない。"""
    cfg = deepcopy(BASE)
    old = PRESETS[preset]
    cfg["preset"] = preset
    cfg["hardware_profile"] = hardware
    cfg["training"].update(deepcopy(HARDWARE_PROFILES[hardware]))
    cfg["extra_context"]["mode"] = extra_context
    if preset != "reference_gemma31":
        cfg["hardware_profile"] = "legacy_single_gpu_starting_values"
        cfg["model"].update(model_id=old["model_id"], backend=old["backend"],
                            quantization=old["quantization"],
                            load_in_4bit=old["quantization"] != "none",
                            max_seq_length=old["max_seq_length"],
                            environment_before_import={})
        cfg["lora"].update(r=old["r"], alpha=old["alpha"], dropout=0.05,
                           finetune_mlp_modules=False)
        cfg["lora"]["resolve_backend_specific_targets_from_source"] = old["source"]
        cfg["external_data"]["enabled"] = False
        cfg["prompt"].update(repeat=1, answer_prefix=None, template_kwargs={})
        cfg["image"].update(max_soft_tokens=None, longest_side_cap=None)
        cfg["image"]["augmentation"]["target"] = ["orig"]
        cfg["data"]["cleaning"].update(mode="precleaned_manifest", required_files=[])
        cfg["data"]["dev_votes"]["enabled"] = False
        cfg["training"].update(
            stages=[{"name": "target", "epochs": 1.0, "learning_rate": old["lr"],
                     "augmentation": ["orig"], "save_adapter": "adapter"}],
            carry_same_adapter_external_to_target=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            per_device_eval_batch_size=1,
            evidence="기존 노트북의 batch1/accumulation8 시작값; reference 하드웨어 배치는 적용 안 함",
        )
        cfg["inference"]["shifts"] = [0]
        cfg["environment_reference"] = {
            "source": "선택한 기존 노트북별 환경 확인 필요", "record_actual_versions_and_freeze": True
        }
        cfg["evidence"].update(
            previous_notebook=old["source"],
            compatibility_note="핵심 설정 보존 프로필; 원본 실행을 그대로 재현하는 실행기는 아님",
        )
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """초기 설정의 내부 일관성만 검사한다. 입력 파일/실행 환경 확인은 별도로 필요하다."""
    split = cfg["data"]["split"]
    if abs(split["train_fraction"] + split["validation_fraction"] - 1.0) > 1e-9:
        raise ValueError("train/validation 분할 비율 합은 1이어야 합니다")
    if cfg["artifacts"]["class_order"] != cfg["data"]["classes"]:
        raise ValueError("데이터와 저장 확률의 클래스 순서가 다릅니다")
    shifts = cfg["inference"]["shifts"]
    if not shifts or len(set(shifts)) != len(shifts) or not set(shifts) <= {0, 1, 2, 3}:
        raise ValueError("shift는 0~3 범위의 중복 없는 목록이어야 합니다")
    if cfg["artifacts"]["logits_reference_shift"] not in shifts:
        raise ValueError("실제 logits 저장용 기준 shift가 추론 목록에 있어야 합니다")
    if cfg["loss"]["supervised_tokens_per_sample"] != 1:
        raise ValueError("현재 기본 loss는 정답 글자 1토큰을 감독합니다")
    stages = cfg["training"]["stages"]
    if cfg["external_data"]["enabled"] and [s["name"] for s in stages] != ["external", "target"]:
        raise ValueError("외부→내부 2단계의 순서가 다릅니다")


def main():
    parser = argparse.ArgumentParser(description="VQA 초기 설정 출력/JSON 저장 — 학습 실행 아님")
    parser.add_argument("--preset", choices=PRESETS, default=ACTIVE_PRESET)
    parser.add_argument("--hardware", choices=HARDWARE_PROFILES, default=ACTIVE_HARDWARE)
    parser.add_argument("--extra-context", choices=["none", "ocr", "dino", "ocr_dino"], default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = build_config(args.preset, args.hardware, args.extra_context)
    rendered = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"설정 저장: {args.output.resolve()} (학습/추론은 실행하지 않음)")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
