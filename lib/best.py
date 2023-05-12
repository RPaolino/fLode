best_hyperparams = {
    'Cora': {
        "num_layers":3,
        "input_dropout":0.15,
        "decoder_layers":1,
        "encoder_layers":1,
        "decoder_dropout":0.4,
        "hidden_channels":64, 
        "sparsity":0,
        "add_self_loops":True,
        "undirected":True, 
        "patience":200,
        "lr":0.0001,
        "weight_decay":0.0005, 
        "norm_ord": "inf",
        "norm_dim":1, 
    },
    'Citeseer': {
        "num_layers":2,
        "input_dropout":0.1,
        "decoder_layers":1,
        "encoder_layers":1,
        "decoder_dropout":0,
        "hidden_channels":64, 
        "sparsity":0,
        "add_self_loops":True,
        "undirected":True, 
        "patience":100,
        "lr":0.001,
        "weight_decay":0.0005,
        "norm_ord":"sum",
        "norm_dim":1
    },
    'Pubmed': {
        "num_layers":3,
        "input_dropout":0.05,
        "decoder_layers":1,
        "encoder_layers":3,
        "decoder_dropout":0.1,
        "hidden_channels":32, 
        "sparsity":0.5,
        "sklearn": True,
        "add_self_loops":True,
        "undirected":True, 
        "patience":100,
        "lr":0.01,
        "weight_decay":0.001,
        "norm_ord":"inf", 
        "norm_dim":1
    },
    'chameleon':{
        'directed': {
            "num_layers":9,
            "input_dropout":0.1,
            "decoder_layers":2,
            "encoder_layers":1,
            "decoder_dropout":0,
            "hidden_channels":64, 
            "sparsity":0.5,
            "add_self_loops":False,
            "undirected":False, 
            "patience":100,
            "lr":0.005,
            "weight_decay":0.001,
            "norm_ord":2,
            "norm_dim":0
        },
        'undirected': {
            "num_layers":4,
            "input_dropout":0.1,
            "decoder_layers":2,
            "encoder_layers":2,
            "decoder_dropout":0,
            "hidden_channels":64,
            "sparsity":0.5,
            "add_self_loops":False,
            "undirected":True,
            "patience":100,
            "lr":0.005,
            "weight_decay":0.001,
            "norm_ord":2,
            "norm_p":0
        }
    },
    'squirrel':{
        'directed': {
            "num_layers":6,
            "input_dropout":0.1,
            "decoder_layers":2,
            "encoder_layers":1,
            "decoder_dropout":0.1,
            "hidden_channels":64, 
            "sparsity":0.5,
            "add_self_loops":False,
            "undirected":False,
            "patience":100,
            "lr":0.0025,
            "weight_decay":0.0005,
            "norm_ord":2,
            "norm_dim":0
        },
        'undirected':{
            "num_layers":6,
            "input_dropout":0.15,
            "decoder_layers":2,
            "encoder_layers":1,
            "decoder_dropout":0.1,
            "hidden_channels":64,
            "sparsity":0.5,
            "add_self_loops":False,
            "undirected":True,
            "patience":100,
            "lr":0.0025,
            "weight_decay":0.0005,
            "norm_ord":2,
            "norm_p":0
        },
    },
    'Actor': {
        'directed': {
            "num_layers":1,
            "input_dropout":0,
            "decoder_layers":2,
            "encoder_layers":2,
            "decoder_dropout":0.1,
            "hidden_channels":256,
            "sparsity":0.7,
            "add_self_loops":False,
            "undirected":False,
            "patience":100,
            "lr":0.00125,
            "weight_decay":0.00075,
            "norm_ord":"sum",
            "norm_p":1
        },
        'undirected': {
            "num_layers":1,
            "input_dropout":0,
            "decoder_layers":2,
            "encoder_layers":2,
            "decoder_dropout":0.1,
            "hidden_channels":256,
            "sparsity":0,
            "add_self_loops":False,
            "undirected":True,
            "patience":100,
            "lr":0.00125,
            "weight_decay":0.00075,
            "norm_ord":"sum",
            "norm_p":1
        }
    }
}