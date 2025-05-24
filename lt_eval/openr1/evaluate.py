# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import random
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    loglikelihood_acc_metric,
    multilingual_extractive_match_metric,
)
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.tasks.default_prompts import mmlu_helm


latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

loglikelihood_acc = loglikelihood_acc_metric()

expr_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(ExprExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

gpqa_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=5,
)


def prompt_fn(line, task_name: str = None):
    """Assumes the model is either prompted to emit \\boxed{answer} or does so automatically"""
    return Doc(
        task_name=task_name,
        query=line["problem"],
        choices=[line["solution"]],
        gold_index=0,
    )


math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[latex_gold_metric],
    version=1,
)

gsm8k = LightevalTaskConfig(
    name="gsm8k",
    suite=["custom"],
    prompt_function=prompt.gsm8k,
    hf_repo="gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=512,
    metric=[expr_gold_metric],
    stop_sequence=None,
    trust_dataset=True,
    version=0,
)

logiqa = LightevalTaskConfig(
    name="logiqa",
    suite=["custom"],
    prompt_function=prompt.logiqa,
    hf_repo="/online1/ycsc_lijt1/lijt1/wpz/Llama_CDG/lt_eval/logiqa_harness",
    hf_subset="logiqa",
    hf_avail_splits=["train", "validation", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=-1,
    metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)
# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(gsm8k)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(logiqa)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))
