import argparse
import dspy
import litellm
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from dspy.teleprompt import GEPA as LLMOptimizer
from sklearn.model_selection import train_test_split
from src.finetuning import judge_metric_bin as judge_metric
from src.finetuning import judge_metric_float
from src.finetuning import PromptOptimizerBase
from src.finetuning import save_prompt
from src.logger_setup import setup_dspy_logger
from src.mask_data import mask_data
from src.prompt import Prompts


litellm.ssl_verify = False

log = setup_dspy_logger(log_file="gepa.log")
log.info("GEPA compile started")

tqdm.pandas()
load_dotenv()

reflection_lm = dspy.LM(
    model="openai//qwen36-27b",
    api_key="empty",
    api_base="https://qwen.k8s.dev002.local/v1",
    temperature=1.0,
)

dspy.configure(lm=reflection_lm)


def build_data(items):
    examples = []
    for item in items:
        subject = item["subject"]
        body_text = item["body_text"]
        body_text = mask_data(body_text)
        example = dspy.Example(
            email=f"{subject} {body_text}",
            expected=item["expected"],
        ).with_inputs("email")
        examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()
    PREDICT_MODE = args.mode

    data = None
    student = None
    metric = None

    columns = ["subject", "body_text", "expected", "source"]
    if PREDICT_MODE == "recipient_kos":
        metric = judge_metric_float
        student = PromptOptimizerBase(Prompts.receiver_kos)
        df = pd.read_excel("./data/eval_data__recipient.xlsx", sheet_name="Sheet1")
        df.query('source == "kos@kos.sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "recipient_nknh":
        metric = judge_metric_float
        student = PromptOptimizerBase(Prompts.receiver_nknh)
        df = pd.read_excel("./data/eval_data__recipient.xlsx", sheet_name="Sheet1")
        df.query('source == "nknh@sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "recipient_mos":
        metric = judge_metric_float
        student = PromptOptimizerBase(Prompts.receiver_mos)
        df = pd.read_excel("./data/eval_data__recipient.xlsx", sheet_name="Sheet1")
        df.query('source == "info@sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "forward_kos":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.forward_kos)
        df = pd.read_excel("./data/eval_data__forward.xlsx")
        df.query('source == "kos@kos.sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "forward_nknh":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.forward_nknh)
        df = pd.read_excel("./data/eval_data__forward.xlsx")
        df.query('source == "nknh@sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "forward_mos":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.forward_mos)
        df = pd.read_excel("./data/eval_data__forward.xlsx")
        df.query('source == "info@sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "isGD":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.isgd)
        df = pd.read_excel("./data/eval_data__isGD.xlsx")
        df["expected"] = df["expected"].astype(str).str.upper()
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "deadline":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.deadline)
        df = pd.read_excel("./data/eval_data__deadline.xlsx")
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "sender":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.sender)
        df = pd.read_excel("./data/eval_data__sender.xlsx")
        df["expected"] = df["expected"].astype(str).str.upper()
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "kis_nknh":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.kis)
        df = pd.read_excel("./data/eval_data__is_kisecm.xlsx")
        df.query('source == "nknh@sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "kis_mos":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.kis_mos)
        df = pd.read_excel("./data/eval_data__is_kisecm.xlsx")
        df.query('source == "info@sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    elif PREDICT_MODE == "kis_kos":
        metric = judge_metric
        student = PromptOptimizerBase(Prompts.kis)
        df = pd.read_excel("./data/eval_data__is_kisecm.xlsx")
        df.query('source == "kos@kos.sibur.ru"', inplace=True)
        data = df[columns].to_dict(orient="records")
    else:
        pass

    if "kis" in PREDICT_MODE:
        train_data, val_data = train_test_split(
            data, test_size=0.2, random_state=52, shuffle=True, stratify=df["expected"]
        )
    else:
        train_data, val_data = train_test_split(
            data, test_size=0.2, random_state=52, shuffle=True
        )

    trainset = build_data(train_data)
    valset = build_data(val_data)

    log.info(f"=========размер trainset: {len(trainset)}")
    log.info(f"=========размер valset: {len(valset)}")

    optimizer = LLMOptimizer(
        metric=metric,
        auto="medium",  # <-- 'light', 'medium', 'heavy'
        num_threads=5,
        track_stats=True,  # True
        use_merge=True,  # <-- True, False
        reflection_minibatch_size=3,
        reflection_lm=reflection_lm,
    )

    optimized_program = optimizer.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )

    raw = optimized_program.generator.predict.signature.instructions

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"./prompts/{stamp}__prompt_{PREDICT_MODE}.md"

    save_prompt(raw, filename)
    log.info(f"Optimized prompt saved to {filename}")


if __name__ == "__main__":
    main()
