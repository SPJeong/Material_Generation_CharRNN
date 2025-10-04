##### model_trainer.py
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.cuda.amp import autocast, GradScaler
from rdkit import Chem

import CONFIG # custom.py
import chemical_feature_extraction  # custom.py
import my_utils # custom.py
import my_tokenizer # custom.py

# parameter
filtered_num = CONFIG.filtered_num
random_pick_num = CONFIG.random_pick_num
model_name = CONFIG.model_name
data_extraction_folder = CONFIG.data_extraction_folder
chemical_feature_extraction_folder = CONFIG.chemical_feature_extraction_folder
tokenizer_type = CONFIG.tokenizer_type  # tokenizer_type_list = ['gpt', 'smilesPE', 'atomwise', 'atomInSmiles']
smiles_column = CONFIG.smiles_column # 'canonical_smiles' or 'deep_smiles' for tokenizing and numericalizing


# load file for tokenizer
file_folder = chemical_feature_extraction_folder
file_name = f'chemical_feature_extraction_len_{filtered_num}_num_{random_pick_num}_scaled_False_ECFP_False_desc_False.csv'
file_raw_path = os.path.join(file_folder, file_name)

if os.path.exists(file_raw_path):
    print(f"Loading existing file from: {file_raw_path}")
    file_raw = pd.read_csv(file_raw_path)

else:
    print(f"File not found. Generating data and saving to: {file_raw_path}")
    file_raw = chemical_feature_extraction.run_feature_extraction(filtered_num= filtered_num,
                                                                  random_pick_num= random_pick_num,
                                                                  data_extraction_folder= data_extraction_folder,
                                                                  ecfp= False,
                                                                  descriptors= False,
                                                                  scale_descriptors= False,
                                                                  ecfp_radius= None,
                                                                  ecfp_nbits= None,
                                                                  chemical_feature_extraction_folder= chemical_feature_extraction_folder,
                                                                  inference_mode= False,
                                                                  new_smiles_list= None)

# processing for canonical_smiles & deep_smiles
X_file_processed = my_utils.process_to_canonical_or_deep_smiles(df= file_raw , smiles_col= 'smiles', deep_smiles= True)

# tokenizer setting
tokenizer = my_tokenizer.SmilesTokenizer(df= X_file_processed, smiles_column= smiles_column, tokenizer_type= tokenizer_type)

### note) can be eliminate all the tokenize part and make it short in def train part by adding tokenizer argument and set it as None.

def validate(my_model, val_loader, loss_fn, device):

    my_model.eval().to(device)
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            with autocast():
                outputs, _ = my_model(inputs, lengths)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

            total_loss += loss.item()
    avg_val_loss = total_loss / len(val_loader)
    return avg_val_loss


def test(my_model, test_loader, loss_fn, tokenizer, device, num_to_generate= 100):

    my_model.eval().to(device)
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets, lengths) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            with autocast():
                outputs, _ = my_model(inputs, lengths)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

            total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)
    print(f"\n=====> Test Loss: {avg_test_loss:.4g} <=====")

    generated_smiles_list = []
    print(f"Generating {num_to_generate} molecules for quality check...")

    with torch.no_grad():
        generated_smiles_list = my_model.generate(tokenizer=tokenizer,
                                                  max_length=50,
                                                  num_return_sequences=num_to_generate)

    valid_smiles_count = 0
    for smiles in generated_smiles_list:
        mol = Chem.MolFromSmiles(smiles) # RDKit 필요
        if mol is not None:
            valid_smiles_count += 1

    valid_percentage = (valid_smiles_count / num_to_generate) * 100

    print(f"Example Generated SMILES: {generated_smiles_list[:5]}")
    print(f"Valid SMILES Percentage (RDKit check): {valid_percentage:.2f}%")

    return avg_test_loss, generated_smiles_list


# Training function with AMP
def train(my_model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, device, num_epochs, output_dir):
    scaler = GradScaler()
    my_model.train().to(device)
    print("=====> Starting Training")

    best_loss = float('inf')
    best_model_path = os.path.join(output_dir, f'{model_name}_model_len_{filtered_num}_num_{random_pick_num}_{tokenizer_type}.pt')
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (inputs, targets, lengths) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs, _ = my_model(inputs, lengths)
                loss = loss_fn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # validation
        avg_val_loss = validate(my_model, val_loader, loss_fn, device)
        my_model.train()

        lr_scheduler.step(avg_val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4g} | Val Loss: {avg_val_loss:.4g}')

        # Check if the current model is the best one and save
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f'New best model found at epoch {epoch + 1} with Validation loss {best_loss:.4g}. Saving model.')

            # Use the refactored save_model function to save the best model
            my_utils.save_model(model=my_model,
                                save_model_path=best_model_path,
                                epoch=epoch + 1,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                best_loss=best_loss)

        # Generating smiles during training for model checking
        if (epoch + 1) % 3 == 0:
            print(f'--- [Generation Check] at Epoch {epoch + 1} ---')
            my_model.eval()
            with torch.no_grad():
                batch_size_for_generate = 1  # how many smiles will be generated during training

                generated_smiles = my_model.generate(tokenizer=tokenizer, max_length=50,
                                                     num_return_sequences=batch_size_for_generate, )
                print(f"Generated SMILES: {generated_smiles}")
            my_model.train()
        global_step += 1
    print("=====> Training is complete")
    return best_loss

