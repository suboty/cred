import torch
from transformers import AutoTokenizer, AutoModel


class BertEmbeddings:
    def __init__(
            self,
            model: str = '100k_REGEX'
    ):

        if '100k_REGEX' in model:
            model_name = 'yzimmermann/BERT-100k_REGEX'
            self.name = 'bert_100k_regex'
        else:
            raise NotImplementedError

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __repr__(self):
        return self.name

    def get_bert_regex(self, strings):
        sentence_embeddings = []
        for string in strings:
            t_input = self.tokenizer(string, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                last_hidden_state = self.model(**t_input, output_hidden_states=True).hidden_states[-1]

            weights_for_non_padding = t_input.attention_mask * torch.arange(start=1,
                                                                            end=last_hidden_state.shape[1] + 1).unsqueeze(0)

            sum_embeddings = torch.sum(last_hidden_state * weights_for_non_padding.unsqueeze(-1), dim=1)
            num_of_none_padding_tokens = torch.sum(weights_for_non_padding, dim=-1).unsqueeze(-1)
            sentence_embedding = sum_embeddings / num_of_none_padding_tokens

            sentence_embeddings.append(
                sentence_embedding.detach().numpy()
            )
        return sentence_embeddings
