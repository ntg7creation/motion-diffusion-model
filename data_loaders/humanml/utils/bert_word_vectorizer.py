# data_loaders/humanml/utils/bert_word_vectorizer.py
import torch
import hashlib
from typing import Tuple, Optional
from transformers import AutoTokenizer, AutoModel


class BertWordVectorizer:
    """
    HuggingFace BERT word vectorizer that returns per-token embeddings for a sentence.
    - Uses [CLS] ... [SEP] with HF tokenizer (no HumanML 'sos/OTHER' tokens).
    - Returns (embeddings[L, D], attn_mask[L]) on CPU (numpy if requested).
    - Caches by text string to save compute.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_len: int = 64,                 # tokens incl. special tokens
        device: str = "cpu",
        output_as: str = "torch",          # "torch" | "numpy"
        fp16: bool = False,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.max_len = max_len
        self.output_as = output_as
        self.fp16 = fp16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        if self.fp16 and self.device.type == "cuda":
            self.model.half()

        self.hidden_size = self.model.config.hidden_size
        self._cache = {}

    def _key(self, text: str):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            token_embs: [L, D]  (torch on CPU)
            attn_mask: [L]     (torch on CPU; 1 for valid tokens)
            sent_len:  int     (sum(attn_mask))
        """
        ck = self._key(text)
        if ck in self._cache:
            token_embs, attn_mask, sent_len = self._cache[ck]
            return token_embs, attn_mask, sent_len

        toks = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}

        outputs = self.model(**toks)  # last_hidden_state: [1, L, D]
        token_embs = outputs.last_hidden_state.squeeze(0)  # [L, D]
        attn_mask = toks["attention_mask"].squeeze(0)      # [L]

        # Move to CPU; keep as torch (default), or convert to numpy below if needed
        token_embs = token_embs.float().cpu()
        attn_mask = attn_mask.cpu()
        sent_len = int(attn_mask.sum().item())

        if self.output_as == "numpy":
            token_embs = token_embs.numpy()
            attn_mask = attn_mask.numpy()

        self._cache[ck] = (token_embs, attn_mask, sent_len)
        return token_embs, attn_mask, sent_len

    def dim(self) -> int:
        return self.hidden_size
