best_hyperparams = {
    'Cora': {
        "num_layers":2,
        "input_dropout":0.,
        "decoder_layers":1,
        "encoder_layers":2,
        "decoder_dropout":0.,
        "hidden_channels":64, 
        "sparsity":0.,
        "self_loops":1.,
        "undirected":True, 
        "patience":200,
        "learning_rate":0.01,
        "weight_decay":0.005, 
        "normalize_features": True,
        "norm_ord": float("inf"),
        "norm_dim":1, 
        "lcc": True
    },
    'Citeseer': {
        "num_layers":2,
        "input_dropout":0.,
        "decoder_layers":1,
        "encoder_layers":1,
        "decoder_dropout":0,
        "hidden_channels":64, 
        "sparsity":0.,
        "self_loops":1.,
        "undirected":True, 
        "patience":100,
        "learning_rate":0.01,
        "weight_decay":0.005,
        "normalize_features": True,
        "norm_ord":float("inf"),
        "norm_dim":1,
        "lcc": True
    },
    'Pubmed': {
        "num_layers":3,
        "input_dropout":0.05,
        "decoder_layers":1,
        "encoder_layers":3,
        "decoder_dropout":0.1,
        "hidden_channels":32, 
        "sparsity":0.7,
        "sklearn": True,
        "self_loops":1.,
        "undirected":True, 
        "patience":100,
        "learning_rate":0.01,
        "weight_decay":0.001,
        "normalize_features": True,
        "norm_ord":float('inf'), 
        "norm_dim":1,
        "lcc": True
    },
    'chameleon':{
        'directed': {
            "num_layers":5,
            "input_dropout":0.,
            "decoder_layers":2,
            "encoder_layers":1,
            "decoder_dropout":0.,
            "hidden_channels":64, 
            "sparsity": 0.,
            "self_loops":0.,
            "undirected":False, 
            "patience":100,
            "learning_rate":0.01,
            "weight_decay":0.001,
            "normalize_features": True,
            "norm_ord":2,
            "norm_dim":0,
            "lcc": True
        },
        'undirected': {
            "num_layers":4,
            "input_dropout":0.,
            "decoder_layers":2,
            "encoder_layers":1,
            "decoder_dropout":0.,
            "hidden_channels":64,
            "sparsity":0.,
            "self_loops":0.,
            "undirected":True,
            "patience":100,
            "learning_rate":0.005,
            "weight_decay":0.001,
            "normalize_features": True,
            "norm_ord":2,
            "norm_p":0,
            "lcc": True
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
            "self_loops":0.,
            "undirected":False,
            "patience":100,
            "learning_rate":0.0025,
            "weight_decay":0.0005,
            "normalize_features": True,
            "norm_ord":2,
            "norm_dim":0,
            "lcc": True
        },
        'undirected':{
            "num_layers":6,
            "input_dropout":0.15,
            "decoder_layers":2,
            "encoder_layers":1,
            "decoder_dropout":0.1,
            "hidden_channels":64,
            "sparsity":0.5,
            "self_loops":0.,
            "undirected":True,
            "patience":100,
            "learning_rate":0.0025,
            "weight_decay":0.0005,
            "normalize_features": True,
            "norm_ord":2,
            "norm_p":0,
            "lcc": True
        },
    },
    'film': {
        'directed':  {
            'learning_rate': 0.001, 
            'weight_decay': 0.0005, 
            'num_layers': 1,
            'sparsity': 0., 
            'encoder_layers': 3, 
            'decoder_layers': 2, 
            'hidden_channels': 256, 
            'input_dropout': 0, 
            'decoder_dropout': 0.1, 
            'norm_ord': 'sum', 
            'norm_dim': 1,
            "sklearn":True,
            "undirected": False,
            "lcc": True
        },
        'undirected': {
            'learning_rate': 0.001, 
            'weight_decay': 0.0005, 
            'num_layers': 1,
            'sparsity': 0., 
            'encoder_layers': 3, 
            'decoder_layers': 2, 
            'hidden_channels': 256, 
            'input_dropout': 0, 
            'decoder_dropout': 0.1, 
            'normalize_features': False,
            'norm_ord': 'sum', 
            'norm_dim': 1,
            'patience': 100,
            "undirected": True,
            "lcc": True
        }
    },
    'Minesweeper': {
        "num_layers":4,
        "decoder_layers":2,
        "encoder_layers":2,
        "hidden_channels":512, 
        "sparsity":0.,
        "sklearn": False,
        "learning_rate":1e-3,
        "weight_decay":0.,
        "undirected":True, 
        "layer_norm": True
    },
    'Tolokers': {
        "num_layers":4,
        "decoder_layers":2,
        "encoder_layers":2,
        "hidden_channels":512, 
        "sparsity":0.,
        "sklearn": False,
        "undirected":True, 
        "learning_rate":1e-3,
        "weight_decay":0.,
        "layer_norm": True
    },
    'Roman-empire': {
        "num_layers":4,
        "decoder_layers":2,
        "encoder_layers":2,
        "hidden_channels":512, 
        "sparsity":0.7,
        "sklearn": True,
        "learning_rate":1e-3,
        "weight_decay":0.,
        "self_loops":1.,
        "normalize_features": True,
        "norm_ord":2,
        "norm_dim":0,
        "layer_norm": True
    }
}