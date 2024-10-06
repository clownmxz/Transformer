# self_made GPT2

from transformers import BertTokenizer, GPT2Model
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn.parameter import Parameter
from tqdm import tqdm
import optuna
from optuna.trial import TrialState



def get_data(filepath):
    token_list = []
    tokenizer = BertTokenizer.from_pretrained("pretrain_model/gpt2-chinese-cluecorpussmall")
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split(",")
            text = "".join(line[1:])
            input = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]
            for id in input_ids[0]:
                token_list.append(id.item())
    f.close()

    token_list = torch.tensor(token_list * 5)
    return token_list

class TextSamplerDataset(Dataset):
    # 初始化函数，传入数据data和序列长度seq_len
    def __init__(self, data, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.data = data

    # 返回数据集的长度
    def __len__(self):
        return self.data.size(0) // self.seq_len

    # 根据索引返回数据
    def __getitem__(self, index):
        # 随机生成一个起始位置
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,)).item()
        # 获取从起始位置开始的序列
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        # 返回序列的前self.seq_len个元素和后self.seq_len个元素
        return full_seq[:-1], full_seq[1:]

class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained("pretrain_model/gpt2-chinese-cluecorpussmall")
        # 其实这下面的操作就相当于是一个GPT2LMHeadModel
        self.lm_head = nn.Linear(768, 21128, bias=False)
        weight = torch.load("model_save/gpt2_lm_head_weight.pth")
        self.lm_head.weight = Parameter(weight)
        self.value_layer = nn.Sequential(
            nn.Linear(768, 1),
            nn.Tanh(),
            nn.Dropout(p=0.1)
        )

    def forward(self, input_tokens):
        outputs = self.gpt2(input_tokens)["last_hidden_state"]
        outputs = nn.Dropout(p=0.1)(outputs)
        logits = self.lm_head(outputs)

        return logits

    @ torch.no_grad()
    def generate(self, generate_num, prompt_token, temperature=1, top_k=0.95):
        # 将prompt_token转换为列表
        prompt_token_new = list(prompt_token)
        # 循环generate_num次
        for i in range(generate_num):
            # 将prompt_token_new转换为tensor，并移动到GPU上
            token_input = torch.tensor([prompt_token_new]).to("cuda")
            # 前向传播，得到logits
            logits = self.forward(token_input)
            # 取出最后一个token的logits
            logits = logits[:, -1, :]
            # 对logits进行softmax归一化，得到概率分布
            probs = torch.softmax(logits / temperature, dim=-1)
            # 根据top_k采样下一个token
            next_token = self.sample_top_p(probs, top_k)
            # 将next_token转换为1维
            next_token = next_token.reshape(-1)
            # 将next_token添加到prompt_token_new中
            prompt_token_new.append(next_token.item())

        # 返回生成的prompt_token_new
        return prompt_token_new

    def sample_top_p(self, probs, top_p):
        # 对probs进行排序，得到排序后的概率和对应的索引
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 计算mask，如果概率减去累积概率大于top_p，则mask为True
        mask = sorted_probs - cumulative_probs > top_p
        # 将mask为True的概率置为0
        sorted_probs[mask] = 0.0

        # 将概率归一化
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        # 从归一化后的概率中采样一个token
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        # 根据采样得到的token，从排序后的索引中找到对应的token
        next_token = torch.gather(sorted_indices, -1, next_token)
        # 返回采样得到的token
        return next_token


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1
    batch_size = 32
    max_length = 128 + 1

    token_list = get_data("datasets/cn_sentiment_cls/ChnSentiCorp.txt")
    train_data = TextSamplerDataset(token_list, max_length)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)


    model = GPT2().to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss_all = 0.
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for token_input, token_target in pbar:
            token_input = token_input.to(device)
            token_target = token_target.to(device)
            logits = model(token_input)
            loss = loss_func(logits.view(-1, logits.size(-1)), token_target.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch + 1}, Train_Loss {loss.item():.4f}")
            loss_all += loss.item()

        loss_all /= len(train_loader)
        trial.report(loss_all, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    torch.save(model.state_dict(), "model_save/self_made_GPT2.pth")

    return loss_all


if __name__ == '__main__':

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
