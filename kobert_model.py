# BERT 모델, Vocabulary 불러오기
import gluonnlp as nlp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from kobert import get_pytorch_kobert_model

device = torch.device('cpu')

# Setting prediction parameters
max_len = 60
batch_size = 64
learning_rate = 5e-5

# Load pre-trained model (weights)
bertmodel, vocab = get_pytorch_kobert_model()


# BERT 모델, Vocabulary 불러오기
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=6,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))


# Load tokenizer from a local directory
kobert_tokenizer = AutoTokenizer.from_pretrained("kobert_tokenizer", use_fast=False)
tok = kobert_tokenizer.tokenize

PATH = './KoBERT/'
kobert_model = BERTClassifier(bertmodel, dr_rate=0.5)

kobert_model.load_state_dict(torch.load(PATH + 'model_state_dict.pt', map_location=device))


def text_predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    kobert_model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = kobert_model(token_ids, valid_length, segment_ids)

        test_eval = []
        logit_list = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            logit_list.append(logits)

            # label_encoder = {"정상": 0, "도움요청": 1, "강도범죄": 2, "강제추행(성범죄)": 3, "절도범죄": 4, "폭력범죄":5}
            if np.argmax(logits) == 0:
                test_eval.append("정상")
            elif np.argmax(logits) == 1:
                test_eval.append("도움요청")
            elif np.argmax(logits) == 2:
                test_eval.append("강도범죄")
            elif np.argmax(logits) == 3:
                test_eval.append("강제추행(성범죄)")
            elif np.argmax(logits) == 4:
                test_eval.append("절도범죄")
            elif np.argmax(logits) == 5:
                test_eval.append("폭력범죄")

        # print(">> 입력하신 내용은 " + test_eval[0])
        return test_eval[0], logit_list[0]
