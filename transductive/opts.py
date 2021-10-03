import os,sys
import argparse

def get_opt():
    parser = argparse.ArgumentParser()

    # GNN
    parser.add_argument('--model', default='sgc', type=str) #gcn gat ppnp sage cheb sgc
    parser.add_argument('--num-hidden', default=32, type=int)
    parser.add_argument('--num-feature', default=745, type=int)
    parser.add_argument('--num-class', default=7, type=int)
    parser.add_argument('--num-layer', default=2, type=int)

    # Dataset
    parser.add_argument('--data-path', default='../data/', type=str, help="data path (dictionary)")
    parser.add_argument('--data-name',  default='cora', type=str, help="data name")#cora citeseer pubmed photo computers
    parser.add_argument('--size-imb-type', default='none', type=str, help="the imbalace type of the training set") #none, step
    parser.add_argument('--train-each', default=20, type=int, help="the training size of each class, used in none imbe type")
    parser.add_argument('--valid-each', default=30, type=int, help="the validation size of each class")
    parser.add_argument('--labeling-ratio', default=0.01, type=float, help="the labeling ratio of the dataset, used in step imb type")
    parser.add_argument('--head-list',  default=[0,1,2], type=int, nargs='+', help="list of the majority class, used in step imb type")
    parser.add_argument('--imb-ratio',  default=1.0, type=float, help="the ratio of the majority class size to the minoriry class size, used in step imb type") 
    
    # Training
    parser.add_argument('--lr', default=0.0075, type=float)
    parser.add_argument('--lr-decay-epoch', default=20, type=int)
    parser.add_argument('--lr-decay-rate', default=0.95, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--least-epoch', default=40, type=int)
    parser.add_argument('--early-stop', default=20, type=int)
    parser.add_argument('--log-path',  default='log.txt', type=str)
    parser.add_argument('--saved-model', default='best-model.pt', type=str)

    # Running
    parser.add_argument('--run-split-num', default=5, type=int, help='run N different split times')
    parser.add_argument('--run-init-num',  default=3, type=int, help='run N different init seeds')

    #Pagerank 
    parser.add_argument('--pagerank-prob', default=0.85, type=float,help="probility of going down instead of going back to the starting position in the random walk")
    parser.add_argument('--ppr-topk', default=-1,type=int)
    
    #ReNode
    parser.add_argument('--renode-reweight', '-rr',   default=0,   type=int,   help="switch of ReNode") # 0 (not use) or 1 (use)
    parser.add_argument('--rn-base-weight',  '-rbw',  default=0.5, type=float, help="the base  weight of renode reweight")
    parser.add_argument('--rn-scale-weight', '-rsw',  default=1.0, type=float, help="the scale weight of renode reweight")

    #Imb_loss
    parser.add_argument('--loss-name',    default="ce",   type=str,   help="the training loss")#ce focal re-weight cb-softmax
    parser.add_argument('--factor-focal', default=2.0,    type=float, help="alpha in Focal Loss")
    parser.add_argument('--factor-cb',    default=0.9999, type=float, help="beta  in CB Loss")


    opt = parser.parse_args()

    opt.data_path = opt.data_path + opt.data_name

    opt.shuffle_seed_list = [i for i in range(opt.run_split_num)]
    opt.seed_list         = [i for i in range(opt.run_init_num) ]

    opt.ppr_file = "{}/{}_ppr.pt".format(opt.data_path,opt.data_name) # the pre-computed Personalized PageRank Matrix
    
    return opt
