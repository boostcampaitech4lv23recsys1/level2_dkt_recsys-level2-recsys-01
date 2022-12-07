import torch

def get_attn_mask(mask):
    mask_pad = (
        torch.BoolTensor(mask == 1).unsqueeze(1).unsqueeze(1)
    )  # (batch_size, 1, 1, max_len)
    mask_time = (
            1 - torch.triu(torch.ones((1, 1, mask.size(1), mask.size(1))), diagonal=1)
    ).bool()  # (batch_size, 1, max_len, max_len)
    mask = (mask_pad & mask_time)

    return mask


def feature_embedding(
        X,
        cat_cols,
        emb_cat_dict,
        cat_comb_proj,
        num_comb_proj,
        device
    ):
    cat_feature = X['cat'].to(device)
    num_feature = X['num'].to(device)

    cat_emb_list = []
    for idx, cat_col in enumerate(cat_cols):
        cat_emb_list.append(emb_cat_dict[cat_col](cat_feature[:, :, idx]))

    cat_emb = torch.cat(cat_emb_list, dim=-1)
    cat_emb = cat_comb_proj(cat_emb)
    num_emb = num_comb_proj(num_feature)
    X = torch.cat([cat_emb, num_emb], -1)

    return X

def feature_one_embedding(
    X,
    cat_comb_proj,
    num_comb_proj,
    emb_cat,
    device
):
    cat_feature = X['cat'].to(device)
    num_feature = X['num'].to(device)

    batch_size, max_seq_len, _ = cat_feature.size()
    cat_emb = emb_cat(cat_feature).view(batch_size, max_seq_len, -1)
    cat_emb = cat_comb_proj(cat_emb)
    num_emb = num_comb_proj(num_feature) 

    X = torch.cat([cat_emb, num_emb], -1)

    return X