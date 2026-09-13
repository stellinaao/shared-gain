tv_vals = {
    "response": ["left", "right"],
    "rewarded": ["incorr", "corr"],
    "block_side": ["left", "right"],
    "response_prev": ["left", "none", "right"],
    "rewarded_prev": ["incorr", "corr"],
    "strategy": ["mf", "mb"],
}

tv_name_map = {
    "response_1": "response_left",
    "response_-1": "response_right",
    "rewarded_0": "rewarded_incorr",
    "rewarded_1": "rewarded_corr",
    "block_side_1": "block_side_left",
    "block_side_-1": "block_side_right",
    "response_prev_1": "response_prev_left",
    "response_prev_0": "response_prev_none",
    "response_prev_-1": "response_prev_right",
    "rewarded_prev_0": "rewarded_prev_incorr",
    "rewarded_prev_1": "rewarded_prev_corr",
    "strategy_-1": "strategy_mf",
    "strategy_1": "strategy_mb",
}

tv_pos_neg = {
    "response": {"pos": "left", "neg": "right"},
    "rewarded": {"pos": "corr", "neg": "incorr"},
    "block_side": {"pos": "left", "neg": "right"},
    "response_prev": {"pos": "left", "neg": "right"},
    "rewarded_prev": {"pos": "corr", "neg": "incorr"},
}
