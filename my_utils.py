##### my_utils.py

import os
import torch
import deepsmiles
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class myChar_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        numericalized_smiles = self.df['numericalize_token'].iloc[index]

        # input, target preparation
        input_tokens = numericalized_smiles[:-1]
        target_tokens = numericalized_smiles[1:]

        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        sequence_length = len(input_tokens)

        return input_tensor, target_tensor, sequence_length


def my_collate_fn(batch, pad_idx=0):
    # separate data from batch
    inputs, targets, lengths = zip(*batch)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_idx)

    return padded_inputs, padded_targets, torch.tensor(lengths, dtype=torch.long)


def process_to_canonical_or_deep_smiles(df, smiles_col, deep_smiles=False):
    """
    Args:
        df (pd.DataFrame): DataFrame.
        smiles_col (str): SMILES column name.
        deep_smiles (bool): default = False.

    Returns:
        pd.DataFrame: Mol 객체, Canonical SMILES, (옵션으로) DeepSMILES가 추가된 DataFrame.
    """

    mol_list = []
    for smiles in tqdm(df[smiles_col], desc="Converting SMILES to Mol"):
        mol_list.append(Chem.MolFromSmiles(smiles))

    df_processed = df.copy()
    df_processed['mol'] = mol_list

    indices_to_drop = df_processed[df_processed['mol'].isnull()].index.tolist()
    if indices_to_drop:
        print(f"Dropping {len(indices_to_drop)} rows due to failed Mol conversion.")
        df_processed = df_processed.drop(indices_to_drop)

    df_processed = df_processed.reset_index(drop=True)  # reset index
    print("\n Starting Canonical SMILES conversion (Mol -> SMILES)...")

    canonical_smiles_list = []
    for mol in tqdm(df_processed['mol'], desc="Converting Mol to Canonical SMILES"):
        canonical_smiles_list.append(Chem.MolToSmiles(mol))

    df_processed['canonical_smiles'] = canonical_smiles_list

    if deep_smiles:
        print("\n Starting DeepSMILES conversion (Canonical SMILES -> DeepSMILES)...")
        try:
            converter = deepsmiles.Converter(rings=True, branches=True)

            deep_smiles_list = []
            for canonical_smiles in tqdm(df_processed['canonical_smiles'], desc="Encoding DeepSMILES"):
                try:
                    encoded = converter.encode(canonical_smiles)
                    deep_smiles_list.append(encoded)
                except Exception as e:
                    print(f"DeepSMILES encoding failed for {canonical_smiles}: {e}. Appending None.")
                    deep_smiles_list.append(None)

            df_processed['deep_smiles'] = deep_smiles_list
            print("DeepSMILES conversion complete.")

        except Exception as e:
            print(f"Warning: DeepSMILES conversion failed completely. Check deepsmiles installation. Error: {e}")

    print("\nProcessing complete. Returning DataFrame.")
    return df_processed


def save_model(model,
               save_model_path,
               epoch=None,
               global_step=None,
               optimizer=None,
               lr_scheduler=None,
               best_loss=None):
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    checkpoint = {'model_state_dict': model.state_dict(), }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if global_step is not None:
        checkpoint['global_step'] = global_step
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss

    torch.save(checkpoint, save_model_path)
    print(f'Saved model checkpoint to {save_model_path}.')

