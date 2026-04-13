param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThroughArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$entryScript = Join-Path $scriptDir "..\slt_data.py"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $entryScript @PassThroughArgs
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $entryScript @PassThroughArgs
    exit $LASTEXITCODE
}

throw "Python 3 is required to run data/scripts/slt_data.py."
