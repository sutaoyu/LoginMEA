import numpy as np
import scipy.sparse as sp
from collections import Counter
import json
import pickle
from tqdm import tqdm


def load_triples_dual(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name, "r"):
        head, r, tail = [int(item) for item in line.split()]
        entity.add(head)
        entity.add(tail)
        rel.add(r + 1)
        triples.append((head, r + 1, tail))
    return entity, rel, triples


def normalize_adj_dual(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def get_matrix(triples, entity, rel):
    ent_size = max(entity) + 1
    rel_size = max(rel) + 1
    print(ent_size, rel_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    for i in range(max(entity) + 1):
        adj_features[i, i] = 1

    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r])
        radj.append([t, h, r + rel_size])
        rel_out[h][r] += 1
        rel_in[t][r] += 1

    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if " ".join([str(h), str(t)]) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(" ".join([str(h), str(t)]))
            r_index.append([count, r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj_dual(adj_features)
    rel_features = normalize_adj_dual(sp.lil_matrix(rel_features))
    return adj_matrix, r_index, r_val, adj_features, rel_features


def load_data_dual(lang):
    import os

    entity1, rel1, triples1 = load_triples_dual(os.path.join(lang, "triples_1"))
    entity2, rel2, triples2 = load_triples_dual(os.path.join(lang, "triples_2"))

    adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(
        triples1 + triples2, entity1.union(entity2), rel1.union(rel2)
    )

    return adj_matrix, np.array(r_index), np.array(r_val), adj_features, rel_features


def loadfile(fn, num=1):
    print("loading a file..." + fn)
    ret = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ids(fn):
    ids = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split("\t")
            ids.append(int(th[0]))
    return ids


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                ent2id[th[1]] = int(th[0])
    return ent2id


def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    for fn in fns:
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                th = line[:-1].split("\t")
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
    return attr


def load_relation(e, KG, topR=1000):
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.0
            rel_mat[o][rel_index_dict[r]] += 1.0
    return np.array(rel_mat)


def load_json_embd(path):
    embd_dict = {}
    with open(path) as f:
        for line in f:
            example = json.loads(line.strip())
            vec = np.array([float(e) for e in example["feature"].split()])
            embd_dict[int(example["guid"])] = vec
    return embd_dict


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    img_embd = np.array(
        [
            img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0])
            for i in range(e_num)
        ]
    )
    print(
        "%.2f%% entities have images, use np.random.normal init"
        % (100 * len(img_dict) / e_num)
    )
    return img_embd


def load_img_zero(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    img_embd = np.array(
        [
            img_dict[i] if i in img_dict else np.zeros(mean.shape[0])
            for i in range(e_num)
        ]
    )
    print(
        "%.2f%% entities have images, use np.random.zeros init"
        % (100 * len(img_dict) / e_num)
    )
    return img_embd


def load_img_new(e_num, path, triples):
    from collections import defaultdict

    img_dict = pickle.load(open(path, "rb"))
    neighbor_list = defaultdict(list)
    for triple in triples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if tail in img_dict:
            neighbor_list[head].append(tail)
        if head in img_dict:
            neighbor_list[tail].append(head)
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    all_img_emb_normal = np.random.normal(mean, std, mean.shape[0])
    img_embd = []
    follow_neirbor_img_num = 0
    follow_all_img_num = 0
    for i in range(e_num):
        if i in img_dict:
            img_embd.append(img_dict[i])
        else:
            if len(neighbor_list[i]) > 0:
                follow_neirbor_img_num += 1
                if i in img_dict:
                    neighbor_list[i].append(i)
                neighbor_imgs_emb = np.array([img_dict[id] for id in neighbor_list[i]])
                neighbor_imgs_emb_mean = np.mean(neighbor_imgs_emb, axis=0)
                img_embd.append(neighbor_imgs_emb_mean)
            else:
                follow_all_img_num += 1
                img_embd.append(all_img_emb_normal)
    print(
        "%.2f%% entities have images," % (100 * len(img_dict) / e_num),
        " follow_neirbor_img_num is {0},".format(follow_neirbor_img_num),
        " follow_all_img_num is {0}".format(follow_all_img_num),
    )
    return np.array(img_embd)


def load_word2vec(path, dim=300):
    """
    glove or fasttext embedding
    """
    print("\n", path)
    word2vec = dict()
    err_num = 0
    err_list = []
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(file.readlines(), desc="load word embedding"):
            line = line.strip("\n").split(" ")
            if len(line) != dim + 1:
                continue
            try:
                v = np.array(list(map(float, line[1:])), dtype=np.float64)
                word2vec[line[0].lower()] = v
            except:
                err_num += 1
                err_list.append(line[0])
                continue
    file.close()
    print("err list ", err_list)
    print("err num ", err_num)
    return word2vec


def load_char_bigram(path):
    ent_names = json.load(open(path, "r"))
    char2id = {}
    count = 0
    for _, name in ent_names:
        for word in name:
            word = word.lower()
            for idx in range(len(word) - 1):
                if word[idx : idx + 2] not in char2id:
                    char2id[word[idx : idx + 2]] = count
                    count += 1
    return ent_names, char2id


def load_word_char_features(node_size, word2vec_path, name_path):
    """
    node_size : ent num
    """
    word_vecs = load_word2vec(word2vec_path)
    ent_names, char2id = load_char_bigram(name_path)

    ent_vec = np.zeros((node_size, 300))
    char_vec = np.zeros((node_size, len(char2id)))
    for i, name in ent_names:
        k = 0
        for word in name:
            word = word.lower()
            if word in word_vecs:
                ent_vec[i] += word_vecs[word]
                k += 1
            for idx in range(len(word) - 1):
                char_vec[i, char2id[word[idx : idx + 2]]] += 1
        if k:
            ent_vec[i] /= k
        else:
            ent_vec[i] = np.random.random(300) - 0.5

        if np.sum(char_vec[i]) == 0:
            char_vec[i] = np.random.random(len(char2id)) - 0.5
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])

    return ent_vec, char_vec
