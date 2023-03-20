# BERT 모델, Vocabulary 불러오기
import gluonnlp as nlp
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

import time


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


device = torch.device('cpu')

# Setting prediction parameters
max_len = 60
batch_size = 64
learning_rate = 5e-5

print("Loading BERT model...")
# Load pre-trained model (weights)
bertmodel, vocab = get_pytorch_kobert_model()

# Load tokenizer from a local directory
# kobert_tokenizer = AutoTokenizer.from_pretrained("kobert_tokenizer", use_fast=False)
# tok = kobert_tokenizer.tokenize
print("Loading BERT tokenizer...")
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

PATH = './KoBERT/'
kobert_model = BERTClassifier(bertmodel, dr_rate=0.5)
kobert_model.load_state_dict(torch.load(PATH + 'model_state_dict.pt', map_location=device))


def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx])) / a).item() * 100


def text_predict(predict_sentence, model=kobert_model):
    print("predictsentence start:", predict_sentence)
    start = time.time()
    text_label = ['regular', 'help', 'robbery', 'sexual', 'theft', 'violence']
    data = [predict_sentence]
    # dataset_another = [data]

    transform = nlp.data.BERTSentenceTransform(tok, max_len, pad=True, pair=False)
    tokenized = transform(data)
    model.eval()

    # print([tokenized[0]])
    # token_ids = torch.tensor([tokenized[0]]).to(device)
    # segment_ids = torch.tensor([tokenized[2]]).to(device)
    token_ids = torch.tensor(np.array([tokenized[0]])).to(device)
    valid_length = [tokenized[1]]
    segment_ids = torch.tensor(np.array([tokenized[2]])).to(device)

    result = model(token_ids, valid_length, segment_ids)
    # print(result)
    idx = result.argmax().cpu().item()
    out_prob = result.detach().cpu().numpy()[0]
    # print(out_prob)
    print("대사의 카테고리는:", text_label[idx])
    print("대사 신뢰도는:", "{:.2f}%".format(softmax(result, idx)))
    end = time.time() - start
    print("text predict 걸린 시간:", end)
    return text_label[idx], out_prob
