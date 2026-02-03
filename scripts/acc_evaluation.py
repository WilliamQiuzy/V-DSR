import json
import os
import argparse
import asyncio
import openai
from tqdm import tqdm
import httpx

from openai import AsyncAzureOpenAI, OpenAIError, AsyncOpenAI
from utils.eval_utils import get_acc_async
from dotenv import load_dotenv
load_dotenv()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs/real_mc_cot")
    parser.add_argument('--eval_dir', type=str, default="processed_outputs")
    parser.add_argument('--log_file', type=str, default="evaluation_results.txt")  # Log file for accuracy
    parser.add_argument(
        '--provider',
        type=str,
        default="openai",
        choices=["openai", "deepseek", "azure"],
        help="LLM provider for evaluation (default: openai).",
    )
    parser.add_argument(
        '--model',
        type=str,
        default="",
        help="Model name for evaluation. For deepseek, use deepseek-chat or deepseek-reasoner.",
    )
    args = parser.parse_args()

    def _build_deepseek_http_client():
        proxy = os.getenv("DEEPSEEK_PROXY")
        ca_bundle = os.getenv("DEEPSEEK_CA_BUNDLE")
        insecure = os.getenv("DEEPSEEK_INSECURE", "").lower() in ("1", "true", "yes")
        timeout = float(os.getenv("DEEPSEEK_TIMEOUT", "30.0"))
        verify = False if insecure else (ca_bundle if ca_bundle else True)
        kwargs = {"verify": verify, "timeout": timeout}
        if proxy:
            try:
                return httpx.AsyncClient(proxies=proxy, **kwargs)
            except TypeError as exc:
                if "proxies" in str(exc):
                    return httpx.AsyncClient(proxy=proxy, **kwargs)
                raise
        return httpx.AsyncClient(**kwargs)

    subdir = os.path.basename(args.output_dir)
    os.makedirs(os.path.join(args.eval_dir, subdir), exist_ok=True)

    log_path = os.path.join(args.eval_dir, args.log_file)

    # Open log file in append mode and record output_dir at the beginning
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\nEvaluating output directory: {args.output_dir}\n")
        log_file.write("=" * 50 + "\n")  # Separator for clarity

        http_client = None
        provider = args.provider.lower()
        try:
            if provider == "deepseek":
                http_client = _build_deepseek_http_client()

            for output_file in os.listdir(args.output_dir):
                print('Now testing:', output_file)
                if not output_file.endswith(".json"):
                    continue

                examples_path = os.path.join(args.output_dir, output_file)
                with open(examples_path, "r", encoding="utf-8") as f:
                    examples = json.load(f)

                eval_file = os.path.join(args.eval_dir, subdir, output_file)

                # Skip if it's already evaluated
                if os.path.exists(eval_file):
                    with open(eval_file, "r", encoding="utf-8") as f:
                        eval_results = json.load(f)
                    if (len(eval_results) == len(examples)
                        and eval_results and eval_results[0]["response"] == examples[0]["response"]):
                        print(f"Skipping {output_file}")
                        continue

                client = None
                if provider == "azure":
                    client = AsyncAzureOpenAI(
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    )
                elif provider == "deepseek":
                    client = AsyncOpenAI(
                        api_key=os.getenv("DEEPSEEK_API_KEY"),
                        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                        http_client=http_client,
                    )
                else:
                    client = AsyncOpenAI(
                        api_key=os.getenv("OPENAI_API_KEY"),
                    )

                model_name = args.model
                if not model_name:
                    if provider == "deepseek":
                        model_name = "deepseek-reasoner"
                    else:
                        model_name = "o4-mini"

                accuracy, outputs = await get_acc_async(
                    examples,
                    client,
                    engine_name=model_name,
                    provider=provider,
                )
                with open(eval_file, "w", encoding="utf-8") as f:
                    json.dump(outputs, f, indent=4, ensure_ascii=False)

                log_entry = f"Accuracy of {output_file}: {accuracy}\n"
                log_file.write(log_entry)  # Write accuracy to log file
                print(log_entry.strip())  # Print the result
        finally:
            if http_client is not None:
                await http_client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
