# from transformers import PfeifferConfig, HoulsbyConfig, ParallelConfig, CompacterConfig
from adapters import DoubleSeqBnConfig,SeqBnConfig,ParBnConfig,CompacterConfig
ADAPTER_MAP = {
    'pfeiffer': SeqBnConfig,
    'houlsby': DoubleSeqBnConfig,
    'parallel': ParBnConfig,
    'compacter': CompacterConfig,
}

