from torch.utils.data import Dataset
from ..config import CFG

class BaseDataset(Dataset):
    def __init__(self, data, config) -> None:
        super().__init__()
        self.data = data
        self.config = config
    
    # 총 데이터의 개수를 리턴
    def __len__(self) -> int:
        return len(self.data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, index: int) -> object:
        row = self.data[index]
        
        # hi
        return
        # return (self.user_id[index], self.item_id[index], self.rating[index])
