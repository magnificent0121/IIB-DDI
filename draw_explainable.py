import pandas as pd
import torch
from rdkit.Chem.Draw import SimilarityMaps
import random
import numpy as np
from utils import create_batch_mask
import os
import argument
import time
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from torch_geometric.data import DataLoader
from utils import get_stats, write_summary, write_summary_total

IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 600, 600
torch.set_num_threads(2)
# df = pd.read_csv(f'data/raw/chemDDI_train.csv', sep=",")
df = pd.read_csv(f'data/raw/ZhangDDI_train.csv', sep=",")


# df = pd.read_csv(f'data/raw/ZhangDDI_train_toy2.csv', sep=",")


def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args, unknown = argument.parse_args()

print("Loading dataset...")
start = time.time()

# Load dataset
train_set = torch.load("./data/processed/{}_train.pt".format(args.dataset))
# valid_set = torch.load("./data/processed/{}_valid.pt".format(args.dataset))
# test_set = torch.load("./data/processed/{}_test.pt".format(args.dataset))

# train_set = torch.load("./data/processed/{}_train_toy2.pt".format(args.dataset))
# valid_set = torch.load("./data/processed/{}_train_toy2.pt".format(args.dataset))
# test_set = torch.load("./data/processed/{}_train_toy2.pt".format(args.dataset))


print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
from models.CGIB import CGIB

device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
model = CGIB(device=device, num_step_message_passing=args.message_passing, dataset=args.dataset).to(device)
# 步骤1：加载保存的模型
checkpoint = torch.load('model_interpretable1.pth', map_location=torch.device('cpu'))

# 提取模型状态字典和额外信息
model.load_state_dict(checkpoint['model_state_dict'], False)

train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
i = 0
z = 0
ls = [83, 284, 339,486,1671,1770,2724,2980,2998,3089]
# base_smile = 'COC(=O)C1=C(C)NC(C)=C(C1C1=CC(=CC=C1)[N+]([O-])=O)C(=O)OCCN(C)CC1=CC=CC=C1=CC=CC=C1'
# new

base_smile = 'NNCCC1=CC=CC=C1'
graph_store_base_path = "./results/explainable/new2/solute/"


base_smile = 'ClCCNC(=O)N(CCCl)N=O'
graph_store_base_path = "./results/explainable/new3/solute/"


base_smile = 'CN[C@@H](C)CC1=CC=CC=C1'
graph_store_base_path = "./results/explainable/new4/solute/"


# new3
for bc, samples in enumerate(train_loader):
    # if i == ls[z]:
    smile_1 = df.iloc[i]["smiles_1"]
    smile_2 = df.iloc[i]["smiles_2"]
    if smile_1==base_smile or smile_2==base_smile:
        print('smile_1: ' + smile_1)
        print('smile_2: ' + smile_2)
        z = z + 1
        masks = create_batch_mask(samples)
        solute_sublist, solvent_sublist = model.get_subgraph(
            [samples[0].to(device), samples[1].to(device), masks[0].to(device), masks[1].to(device)], bottleneck=True)
        if not os.path.exists(graph_store_base_path + str(i)):
            os.makedirs(graph_store_base_path + str(i))

        mol = Chem.MolFromSmiles(smile_1)

        solu_atom_num = mol.GetNumAtoms()
        mol.RemoveAllConformers()
        print(solute_sublist)
        for k in range(len(solute_sublist)):
            for p in range(len(solute_sublist[k])):
                if solute_sublist[k][p] < 0.9:
                    solute_sublist[k][p] = 0
                elif solute_sublist[k][p] < 0.99:
                    solute_sublist[k][p] = (solute_sublist[k][p] - 0.9) * 10
        for j in range(len(solute_sublist)):
            solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, solute_sublist[j].cpu().detach().numpy(),
                                                                     colorMap='RdBu',
                                                                     alpha=0.05,
                                                                     size=(200, 200))
            solvent_fig.savefig(graph_store_base_path + str(i) + "/solute{}.png".format(j),
                                bbox_inches='tight', dpi=600)



        solute_sublist, solvent_sublist = model.get_subgraph(
            [samples[1].to(device), samples[0].to(device), masks[1].to(device), masks[0].to(device)], bottleneck=True)
        if not os.path.exists(graph_store_base_path + str(i)):
            os.makedirs(graph_store_base_path + str(i))
        mol = Chem.MolFromSmiles(smile_2)

        solu_atom_num = mol.GetNumAtoms()
        mol.RemoveAllConformers()
        print(solute_sublist)
        for k in range(len(solute_sublist)):
            for p in range(len(solute_sublist[k])):
                if solute_sublist[k][p] < 0.9:
                    solute_sublist[k][p] = 0
                elif solute_sublist[k][p] < 0.99:
                    solute_sublist[k][p] = (solute_sublist[k][p] - 0.9) * 10
        for j in range(len(solute_sublist)):
            solvent_fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, solute_sublist[j].cpu().detach().numpy(),
                                                                     colorMap='RdBu',
                                                                     alpha=0.05,
                                                                     size=(200, 200))
            solvent_fig.savefig(graph_store_base_path + str(i) + "/solvent{}.png".format(j),
                                bbox_inches='tight', dpi=600)

    i = i + 1
# print(1)


