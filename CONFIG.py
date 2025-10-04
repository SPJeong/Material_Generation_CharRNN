##### CONFIG.py
import torch


filtered_num = 30 # filtering num of SMILES
random_pick_num = 100000 # num_pick
data_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\data_extraction"
chemical_feature_extraction_folder = fr"C:\Users\wisdo\polyOne_Data_Set\chemical_feature_extraction"
model_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\models"
plot_save_folder = fr"C:\Users\wisdo\polyOne_Data_Set\plot"

batch_size = 128
learning_rate = 3e-4
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

embedding_dim = 300
hidden_dim = 512
num_layers = 2
dropout = 0.2
padding_value = 0

ROnPlateauLR_mode = 'min'
ROnPlateauLR_factor = 0.2
ROnPlateauLR_patience = 5

model_name = 'CharRNN'
tokenizer_type ='atomwise' # tokenizer_type_list = ['gpt', 'smilesPE', 'atomwise', 'atomInSmiles']
smiles_column = 'canonical_smiles' # 'canonical_smiles' or 'deep_smiles' for tokenizing and numericalizing
