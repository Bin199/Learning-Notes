import torch
import numpy as np


# from raw text
class Model(object):
    def __init__(self, model_name_or_path: str, device: str = None, pooler = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
    
    def encode(self, sentence: Union[str, List[str]], device: str = None, normalize_to_unit: bool = True,
               keepdim: bool = False, batch_size: int = 64,
               max_length: int = 128) -> Union[ndarray, Tensor]:
        target_device = self.device if device is None else device
        self.model = self.model.to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True
        embedding_list = []
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in tqdm(range(total_batch)):
                inputs = self.tokenizer(
                    sentence[batch_id * batch_size: (batch_id + 1)*batch_size],
                    padding = True,#填充
                    Truncation = True,#截断
                    max_length = max_length,#截断的最大长度
                    return_tensors = "pt" #返回pytorch张量 如果是"tf",则返回tensorflow张量
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict = True)
                if self.pooler == 'cls':
                    embeddings = outputs.pooler_output
                elif self.pooler == 'cls_before_pooler':
                    embeddings = outputs.last_hidden_state[:,0]
                else:
                    raise NotImplementedError
                if normalize_to_list:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim = True)
                embedding_list.append(embeddings.cpu())
            embeddings = torch.cat(embedding_list, 0)
            
            if single_sentence and not keepdim:
                embeddings = embeddings[0]
            if return_numpy and not isinstance(embeddings, ndarray):
                return embeddings.numpy()
            return embeddings
        
# from digital text
class Model1(object):
    def __init__(self, args):
        super(Model,self).__init__()
        
        self.args = args
        self.embed = nn.Embedding(args.embed_num, args.embed_dim, padding_idx = 1)
        self.embed.weight.data.cpoy_(torch.from_numpy(vectors))
        
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.embed.weight.data[0], -0.05, 0.05)
       
        # <pad> vector is initialized as zero padding
        nn.init.constant_(self.embed.weight.data[1], 0)
    def forward(x):
        return x

model = Model1(args)
def encode(self, batch):
    embed = model.embed(x)
    return embed