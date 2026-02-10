from adaptivocab_expand import expand_and_save, ExpansionOptions


if __name__ == "__main__":
    opts = ExpansionOptions(alpha=2.0, device_map="auto", dtype="auto", special_token_strategy="qwen3", verify=True)
    report = expand_and_save(
        base_model="Qwen/Qwen3-8B",
        base_tokenizer="Qwen/Qwen3-8B",
        expanded_tokenizer="DopeorNope/FFT-expanded-naive",
        output_dir="./qwen3-8b-fft-aligned",
        options=opts,
    )
    print(report)
