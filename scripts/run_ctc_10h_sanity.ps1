param(
    [string]$DataRoot = "dataset/datasets/librispeech/LibriSpeech",
    [string]$CmvnPath = "outputs/causal_specunit/targets_960h_c8/cmvn.pt",
    [string]$TokenizerPath = "dataset/bpe128.model",
    [string]$SslCheckpoint = "outputs/causal_specunit/pretrain_ssl_100k_c8/checkpoint_step100000/checkpoint.pt",
    [string[]]$TrainSplits = @("train-clean-100"),
    [string]$EvalSplit = "dev-other",
    [double]$Hours = 10.0,
    [int]$Seed = 42,
    [int]$Epochs = 150,
    [int]$BatchSize = 16,
    [int]$GradAccumSteps = 4,
    [int]$EvalBatchSize = 16,
    [int]$Workers = 2,
    [int]$DataloaderTimeout = 120,
    [string]$Variant = "xs",
    [switch]$SkipSsl,
    [switch]$SkipScratch
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DataRoot)) {
    throw "Missing data root: $DataRoot"
}
if (-not (Test-Path $CmvnPath)) {
    throw "Missing CMVN: $CmvnPath"
}
if (-not (Test-Path $TokenizerPath)) {
    throw "Missing tokenizer: $TokenizerPath"
}
if (-not $SkipSsl -and -not (Test-Path $SslCheckpoint)) {
    throw "Missing SSL checkpoint: $SslCheckpoint"
}

New-Item -ItemType Directory -Force -Path "outputs/causal_specunit" | Out-Null

$common = @(
    "-m", "CausalSpecUnit.train_ctc",
    "--data-root", $DataRoot,
    "--cmvn-path", $CmvnPath,
    "--tokenizer-path", $TokenizerPath,
    "--variant", $Variant,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--grad-accum-steps", "$GradAccumSteps",
    "--eval-batch-size", "$EvalBatchSize",
    "--eval-split", $EvalSplit,
    "--eval-every", "1",
    "--workers", "$Workers",
    "--dataloader-timeout", "$DataloaderTimeout",
    "--train-subset-hours", "$Hours",
    "--train-subset-seed", "$Seed",
    "--progress", "off",
    "--log-every", "0",
    "--save-every", "10"
)
$common += "--train-splits"
$common += $TrainSplits

Write-Host "PyTorch environment:"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('devices', torch.cuda.device_count())"

if (-not $SkipSsl) {
    $sslOutput = "outputs/causal_specunit/ctc_ssl_10h_150ep_c8"
    Write-Host "Starting 10h SSL fine-tune sanity run -> $sslOutput"
    $sslArgs = $common + @(
        "--ssl-checkpoint", $SslCheckpoint,
        "--output-dir", $sslOutput,
        "--lr", "1e-3",
        "--encoder-lr", "3e-4",
        "--head-lr", "1e-3"
    )
    python @sslArgs
}

if (-not $SkipScratch) {
    $scratchOutput = "outputs/causal_specunit/ctc_scratch_10h_150ep_c8"
    Write-Host "Starting 10h scratch sanity run -> $scratchOutput"
    $scratchArgs = $common + @(
        "--output-dir", $scratchOutput,
        "--lr", "2e-3"
    )
    python @scratchArgs
}

Write-Host "Done. Metrics:"
Write-Host "  outputs/causal_specunit/ctc_ssl_10h_150ep_c8/ctc_metrics.jsonl"
Write-Host "  outputs/causal_specunit/ctc_scratch_10h_150ep_c8/ctc_metrics.jsonl"
