import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from transformers import AutoModel


class Classification_head_v2(nn.Module):
    def __init__(self, input_size, in_size, output_size, dropout):
        super(Classification_head_v2, self).__init__()
        self.dance = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, in_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_size),
        )
    
    def forward(self, x):
        return self.dance(x)


class Classification_head(nn.Module):
    def __init__(self, input_size, in_size, output_size, dropout):
        super(Classification_head, self).__init__()
        self.dance = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, in_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_size, output_size)
        )
    
    def forward(self, x):
        return self.dance(x)


class NER_NET(nn.Module):
    def __init__(self, args):
        super(NER_NET, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_path)
        self.classification = Classification_head(args.l_model, args.l_model, args.ner_class, args.dropout)
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        terminal = self.classification(x[:, 1:-1, :])
        return terminal


class SC_NET(nn.Module):
    def __init__(self, args):
        super(SC_NET, self).__init__()
        self.bert = AutoModel.from_pretrained(args.pretrain_path)
        self.classification = Classification_head(args.l_model, args.l_model, args.sc_class, args.dropout)
    
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        terminal = self.classification(x[:, 0, :])
        return terminal


# class BiLSTM_CRF(nn.Module):

#     def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
#         super(BiLSTM_CRF, self).__init__()
#         self.embedding_dim = embedding_dim    # word embedding dim
#         self.hidden_dim = hidden_dim          # Bi-LSTM hidden dim
#         self.vocab_size = vocab_size          
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix)

#         self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
#                             num_layers=1, bidirectional=True)

#         # 将BiLSTM提取的特征向量映射到特征空间，即经过全连接得到发射分数
#         self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

#         # 转移矩阵的参数初始化，transitions[i,j]代表的是从第j个tag转移到第i个tag的转移分数
#         self.transitions = nn.Parameter(
#             torch.randn(self.tagset_size, self.tagset_size))

#         # 初始化所有其他tag转移到START_TAG的分数非常小，即不可能由其他tag转移到START_TAG
#         # 初始化STOP_TAG转移到所有其他tag的分数非常小，即不可能由STOP_TAG转移到其他tag
#         self.transitions.data[tag_to_ix[START_TAG], :] = -10000
#         self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

#         self.hidden = self.init_hidden()

#     def init_hidden(self):
#         # 初始化LSTM的参数
#         return (torch.randn(2, 1, self.hidden_dim // 2),
#                 torch.randn(2, 1, self.hidden_dim // 2))
    
#     def _get_lstm_features(self, sentence):
#         # 通过Bi-LSTM提取特征
#         self.hidden = self.init_hidden()
#         embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
#         lstm_out, self.hidden = self.lstm(embeds, self.hidden)
#         lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
#         lstm_feats = self.hidden2tag(lstm_out)
#         return lstm_feats
    
#     def _score_sentence(self, feats, tags):
#         # 计算给定tag序列的分数，即一条路径的分数
#         score = torch.zeros(1)
#         tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
#         for i, feat in enumerate(feats):
#             # 递推计算路径分数：转移分数 + 发射分数
#             score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
#         score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
#         return score

#     def _forward_alg(self, feats):
#         # 通过前向算法递推计算
#         init_alphas = torch.full((1, self.tagset_size), -10000.)
#         # 初始化step 0即START位置的发射分数，START_TAG取0其他位置取-10000
#         init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

#         # 将初始化START位置为0的发射分数赋值给previous
#         previous = init_alphas

#         # 迭代整个句子
#         for obs in feats:
#             # 当前时间步的前向tensor
#             alphas_t = []
#             for next_tag in range(self.tagset_size):
#                 # 取出当前tag的发射分数，与之前时间步的tag无关
#                 emit_score = obs[next_tag].view(1, -1).expand(1, self.tagset_size)
#                 # 取出当前tag由之前tag转移过来的转移分数
#                 trans_score = self.transitions[next_tag].view(1, -1)
#                 # 当前路径的分数：之前时间步分数 + 转移分数 + 发射分数
#                 next_tag_var = previous + trans_score + emit_score
#                 # 对当前分数取log-sum-exp
#                 alphas_t.append(log_sum_exp(next_tag_var).view(1))
#             # 更新previous 递推计算下一个时间步
#             previous = torch.cat(alphas_t).view(1, -1)
#         # 考虑最终转移到STOP_TAG
#         terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
#         # 计算最终的分数
#         scores = log_sum_exp(terminal_var)
#         return scores


#     def _viterbi_decode(self, feats):
#         backpointers = []

#         # 初始化viterbi的previous变量
#         init_vvars = torch.full((1, self.tagset_size), -10000.)
#         init_vvars[0][self.tag_to_ix[START_TAG]] = 0

#         previous = init_vvars
#         for obs in feats:
#             # 保存当前时间步的回溯指针
#             bptrs_t = []
#             # 保存当前时间步的viterbi变量
#             viterbivars_t = []  

#             for next_tag in range(self.tagset_size):
#                 # 维特比算法记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
#                 # 并不取决与当前tag的发射分数
#                 next_tag_var = previous + self.transitions[next_tag]
#                 best_tag_id = argmax(next_tag_var)
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
#             # 更新previous，加上当前tag的发射分数obs
#             previous = (torch.cat(viterbivars_t) + obs).view(1, -1)
#             # 回溯指针记录当前时间步各个tag来源前一步的tag
#             backpointers.append(bptrs_t)

#         # Transition to STOP_TAG
#         # 考虑转移到STOP_TAG的转移分数
#         terminal_var = previous + self.transitions[self.tag_to_ix[STOP_TAG]]
#         best_tag_id = argmax(terminal_var)
#         path_score = terminal_var[0][best_tag_id]

#         # 通过回溯指针解码出最优路径
#         best_path = [best_tag_id]
#         # best_tag_id作为线头，反向遍历backpointers找到最优路径
#         for bptrs_t in reversed(backpointers):
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # 去除START_TAG
#         start = best_path.pop()
#         assert start == self.tag_to_ix[START_TAG]  # Sanity check
#         best_path.reverse()
#         return path_score, best_path

#     def neg_log_likelihood(self, sentence, tags):
#         # CRF损失函数由两部分组成，真实路径的分数和所有路径的总分数。
#         # 真实路径的分数应该是所有路径中分数最高的。
#         # log真实路径的分数/log所有可能路径的分数，越大越好，构造crf loss函数取反，loss越小越好
#         feats = self._get_lstm_features(sentence)
#         forward_score = self._forward_alg(feats)
#         gold_score = self._score_sentence(feats, tags)
#         return forward_score - gold_score

#     def forward(self, sentence):
#         # 通过BiLSTM提取发射分数
#         lstm_feats = self._get_lstm_features(sentence)

#         # 根据发射分数以及转移分数，通过viterbi解码找到一条最优路径
#         score, tag_seq = self._viterbi_decode(lstm_feats)
#         return score, tag_seq

