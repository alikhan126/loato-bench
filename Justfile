data:
    uv run promptguard data download
    uv run promptguard data harmonize
    uv run promptguard data split

embed:
    uv run promptguard embed --all

sweep:
    uv run promptguard sweep --all

train-standard:
    uv run promptguard train --all --experiment standard_cv

train-loato:
    uv run promptguard train --all --experiment loato

train-transfer:
    uv run promptguard train --all --experiment direct_indirect
    uv run promptguard train --all --experiment crosslingual

analyze:
    uv run promptguard analyze features --all
    uv run promptguard analyze llm-baseline --samples 500
    uv run promptguard analyze report

all: data embed sweep train-standard train-loato train-transfer analyze
