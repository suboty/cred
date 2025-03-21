import os

import torch
from transformers import AutoTokenizer, AutoModel

from logger import logger


class BertEmbeddings:
    def __init__(
            self,
            model: str = 'bert_base_uncased'
    ):

        if 'bert_base_uncased' in model:
            model_name = 'bert-base-uncased'
            self.name = 'bert_base_uncased'
        elif 'codebert_base' in model:
            model_name = 'microsoft/codebert-base'
            self.name = 'bert_base_code'
        elif 'modernbert_base' in model:
            model_name = 'answerdotai/ModernBERT-base'
            self.name = 'bert_base_modern'
        else:
            raise NotImplementedError

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        logger.warning(
            f'This model {self.name} has max length: {self.tokenizer.model_max_length}'
        )

        os.environ['TOKENIZERS_PARALLELISM'] = 'True'

    def __repr__(self):
        return self.name

    def get_bert_regex(self, strings, dialects):
        sentence_embeddings = []
        new_dialects = []
        for i, string in enumerate(strings):
            t_input = self.tokenizer(
                string,
                padding=True,
                return_tensors="pt"
            )

            try:
                with torch.no_grad():
                    last_hidden_state = self.model(**t_input, output_hidden_states=True).hidden_states[-1]

                weights_for_non_padding = t_input.attention_mask * torch.arange(start=1,
                                                                                end=last_hidden_state.shape[
                                                                                        1] + 1).unsqueeze(0)

                sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
                num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
                sentence_embedding = sum_embeddings / num_of_none_padding_tokens

                sentence_embeddings.append(
                    sentence_embedding.detach().numpy()
                )
                new_dialects.append(
                    dialects[i]
                )
            except Exception as e:
                logger.warning(
                    f'Warning! Error <{e}> while making tensor with expression with length: {len(string)}'
                )
        return sentence_embeddings, new_dialects
