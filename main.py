from utils.data_utils import NerDataSet, NerDataLoader
from config.basic_config import configs

if __name__ == "__main__":
    loader = NerDataLoader(mode='train', batch_size=1,
                           drop_last=False, max_length=50, shuffle=False, logger=None,
                           data_dir="./data", num_workers=8, label2id=configs['tag2id'])
    loader = loader.get_dataloader()
    for batch in loader:
        temp = batch
        # print(len(temp))
        print(temp)
        break
    # dataset = loader._get_dataset()
    # features = dataset[0]
    # print(features)
    # from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=128, shuffle=False)
    # print(len(loader))
    # print(len(dataset))
    # for batch in loader:
    #     print(batch)
    #     break