import torch
import torchvision
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch.optim.lr_scheduler import StepLR 
from dataload import MP4Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import logging
from model import TwoTransformerClassifier
import argparse
from plot import plot_and_save
import warnings
warnings.simplefilter('ignore')
def parse_arguments():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--train", action="store_true", help="Whether to train the model (default: True)")
    parser.add_argument("--test", action="store_true", help="Whether to test the model (default: True)")
    parser.add_argument("--input_size", type=int, default=512, help="Input size for the model (default: 512)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers for the model (default: 2)")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads for the model (default: 4)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model (default: 0.1)")
    parser.add_argument("--load_path", type=str, default=None, help="Path to load the model (default: 'None')")#"save/Video_Fusion/model/best_model.pt"
    parser.add_argument("--save_path", type=str, default="save", help="Path to save the model (default: 'save')")
    parser.add_argument("--traindataset", type=str, default="Video_Fusion", choices=["Video_Fusion", "Aphantasia", "ModelZero", "T2VSynthesis","Tune-a-Video"], help="Path to the training dataset")
    parser.add_argument("--testdataset", type=str, default="Video_Fusion", choices=["Video_Fusion", "Aphantasia", "ModelZero", "T2VSynthesis","Tune-a-Video"], help="Path to the testing dataset")
    parser.add_argument("--lr", type=float, default=1e-7,  help="Learning rate for optimization")
    parser.add_argument("--epoch", type=int, default=50,  help="Number of epochs for training")
    args = parser.parse_args()

    return args

def train():
    # 初始化参数
    args = parse_arguments()
    traindataset = args.traindataset
    testdataset = args.testdataset
    train = args.train
    test = args.test
    input_size = args.input_size
    num_layers = args.num_layers
    num_heads = args.num_heads
    dropout = args.dropout
    save_path = os.path.join(args.save_path,traindataset)
    load_path = args.load_path
    model_path=os.path.join(save_path,"model")
    log_path=os.path.join(save_path,"log")
    lr = args.lr
    epoch = args.epoch
    os.makedirs(model_path, exist_ok=True)
    hidden_size = input_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始化log文件
    if train:
        if os.path.exists(os.path.join(log_path,"training.log")):
            os.remove(os.path.join(log_path,"training.log"))
    os.makedirs(log_path, exist_ok=True)
    log_path=os.path.join(save_path,"log","training.log")
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Learning Rate: %f", lr)
    logging.info("Epoch: %d", epoch)
    logging.info("Train Dataset: %s", traindataset)
    logging.info("Test Dataset: %s", testdataset)
    logging.info("Train: %s", train)
    logging.info("Test: %s", test)
    logging.info("Input Size: %d", input_size)
    logging.info("Number of Layers: %d", num_layers)
    logging.info("Number of Heads: %d", num_heads)
    logging.info("Dropout: %.2f", dropout)
    logging.info("Save Path: %s", save_path)
    logging.info('device: %s', device)
    
    train_dataset = MP4Dataset('data/npy/'+traindataset+'/train')
    test_dataset = MP4Dataset('data/npy/'+testdataset+'/test')
    val_dataset = MP4Dataset('data/npy/'+traindataset+'/val')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    model = TwoTransformerClassifier(input_size, hidden_size, num_layers, num_heads, dropout, 1).to(device)
    if load_path:
        model = torch.load(load_path)
    criterion = nn.BCELoss()  
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)  
    if train:
        train_loss=[]
        train_ap=[]
        train_acc=[]
        val_acc=[]
        val_ap=[]
        val_losses=[]
        for epoch in range(epoch): 
            model.train()  
            total_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}, Train Loss: ", unit="batch")
            acc = []
            ap = []
            for data, label in progress_bar:
                optimizer.zero_grad()  
                data, label = data.to(device), label.to(device)
                output = model(data) 
                predicted = (output > 0.5).float() 
                loss = criterion(output, torch.unsqueeze(label.float(), dim=-1)) 
                acc.append(accuracy_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy()))
                ap.append((average_precision_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy(), pos_label=0)+average_precision_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy(), pos_label=1))/2)
                loss.backward() 
                optimizer.step()  
                scheduler.step()
                total_loss += loss.item() * data.size(0) 
                progress_bar.set_postfix({"loss": loss.item()})
            train_accuracy = np.mean(np.array(acc))
            average_loss = total_loss / len(train_loader.dataset)
            train_acc.append(train_accuracy)
            train_ap.append(np.mean(np.array(ap)))
            train_loss.append(average_loss)
            plot_and_save(train_acc,'Train Accuracy',save_path+'/train_acc.png')
            plot_and_save(train_ap,'Train Ap',save_path+'/train_ap.png')
            plot_and_save(train_loss,'Train Loss',save_path+'/train_loss.png')
            logging.info(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.4f}, Train Ap: {np.mean(np.array(ap)):.4f}, Train Loss: {average_loss:.4f}")
            print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.4f}, Train Ap: {np.mean(np.array(ap)):.4f}, Train Loss: {average_loss:.4f}")

            model.eval()  
            total_val_loss = 0.0
            best_val_accuracy = 0
            with torch.no_grad():  
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}, Val Loss: ", unit="batch")
                acc = []
                ap = []
                for data, label in progress_bar:
                    data, label = data.to(device), label.to(device)
                    output = model(data) 
                    predicted = (output > 0.5).float() 
                    acc.append(accuracy_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy()))
                    ap.append((average_precision_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy(), pos_label=0)+average_precision_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy(), pos_label=1))/2)
                    correct = (predicted == label).sum().item()  
                    val_loss = criterion(output, torch.unsqueeze(label.float(), dim=-1))  
                    total_val_loss += val_loss.item() * data.size(0)  
                    progress_bar.set_postfix({"val_loss": val_loss.item()})
                average_val_loss = total_val_loss / len(val_loader.dataset)
                val_accuracy = np.mean(np.array(acc))
                val_acc.append(val_accuracy)
                val_ap.append(np.mean(np.array(ap)))
                val_losses.append(average_val_loss)
                plot_and_save(val_acc,'val Accuracy',save_path+'/val_acc.png')
                plot_and_save(val_ap,'val Ap',save_path+'/val_ap.png')
                plot_and_save(val_losses,'val Loss',save_path+'/val_loss.png')
                logging.info(f"Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}, Val Ap: {np.mean(np.array(ap)):.4f}, Val Loss: {average_val_loss:.4f}")
                print(f"Epoch {epoch+1}, Val Accuracy: {val_accuracy:.4f}, Val Ap: {np.mean(np.array(ap)):.4f}, Val Loss: {average_val_loss:.4f}")
                if best_val_accuracy<val_accuracy :
                    torch.save(model,os.path.join(model_path,'best_model.pt'))
    if test:
        model.eval() 
        total_test_loss = 0.0
        with torch.no_grad():  
            acc = []
            ap = []
            progress_bar = tqdm(test_loader, desc=f"Test Loss: ", unit="batch")
            for data, label in progress_bar:
                data, label = data.to(device), torch.unsqueeze(label.float(), dim=-1).to(device)
                output = model(data)  
                predicted = (output > 0.5).float() 
                acc.append(accuracy_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy()))
                ap.append((average_precision_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy(), pos_label=0)+average_precision_score(predicted.cpu().detach().numpy(), label.cpu().detach().numpy(), pos_label=1))/2)
                test_loss = criterion(output, label)  
                total_test_loss += test_loss.item() * data.size(0)  
                progress_bar.set_postfix({"test_loss": test_loss.item()})
                
        average_test_loss = total_test_loss / len(test_loader.dataset)
        logging.info(f"Test Accuracy: {np.mean(np.array(acc)):.4f},Test AP: {np.mean(np.array(ap)):.4f}, Test Loss: {average_test_loss:.4f}")
        print(f"Test Accuracy: {np.mean(np.array(acc)):.4f},Test AP: {np.mean(np.array(ap)):.4f}, Test Loss: {average_test_loss:.4f}")
    
if __name__ == "__main__":
    train()