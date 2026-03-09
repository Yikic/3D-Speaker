# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import scipy
import sklearn
from sklearn.cluster._kmeans import k_means
from sklearn.metrics.pairwise import cosine_similarity

import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

# try:
#     import umap, hdbscan
# except ImportError:
#     raise ImportError(
#         "Package \"umap\" or \"hdbscan\" not found. \
#         Please install them first by \"pip install umap-learn hdbscan\"."
#         )


class SpectralCluster:
    """A spectral clustering method using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=10, pval=0.02, min_pnum=6, oracle_num=None):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.min_pnum = min_pnum
        self.pval = pval
        self.k = oracle_num

    def __call__(self, X, **kwargs):
        pval = kwargs.get('pval', None)
        oracle_num = kwargs.get('speaker_num', None)
        constraint_matrix = kwargs.get('constraint_matrix', None)

        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat, pval)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Constraint propagation
        if constraint_matrix is not None:
            optim_constraint_matrix = self.e2cp_propagation(sym_prund_sim_mat, constraint_matrix)
            optim_sim_mat = self.adjust_similarity_matrix(sym_prund_sim_mat, optim_constraint_matrix)
        else:
            optim_sim_mat = sym_prund_sim_mat

        # Laplacian calculation
        laplacian = self.get_laplacian(optim_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def e2cp_propagation(self, W, Z, alpha=0.8, method='closed_form', max_iter=100, tol=1e-6):
        """
        Exhaustive and Efficient Constraint Propagation (E2CP) 实现
        
        参数:
        W: np.ndarray, 相似度矩阵 (N x N)
        Z: np.ndarray, 初始约束矩阵 (N x N), +1代表Must-link, -1代表Cannot-link
        alpha: float, 传播参数 (0 < alpha < 1)
        method: str, 'closed_form' (直接求逆) 或 'iterative' (迭代传播)
        max_iter: int, 迭代模式下的最大迭代次数
        tol: float, 迭代收敛阈值
        
        返回:
        F_star: np.ndarray, 传播后的约束矩阵
        """
        N = W.shape[0]
        
        # 1. 计算归一化矩阵 L_bar (论文中定义为 D^-1/2 * W * D^-1/2) [cite: 586]
        d = np.sum(W, axis=1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_bar = D_inv_sqrt @ W @ D_inv_sqrt
        
        I = np.eye(N)
        
        if method == 'closed_form':
            # --- 方式 A: 闭式解 (Closed-form Solution) ---
            # 公式: F* = (1-alpha)^2 * (I - alpha*L_bar)^-1 * Z * (I - alpha*L_bar)^-1 
            inv_mat = np.linalg.inv(I - alpha * L_bar)
            F_star = (1 - alpha)**2 * (inv_mat @ Z @ inv_mat)
            return F_star

        elif method == 'iterative':
            # --- 方式 B: 迭代法 (Efficient Iterative Method) ---
            # 第一步：垂直传播 (Vertical Propagation) 
            Fv = Z.copy()
            for i in range(max_iter):
                Fv_next = alpha * (L_bar @ Fv) + (1 - alpha) * Z
                if np.linalg.norm(Fv_next - Fv, ord='fro') < tol:
                    Fv = Fv_next
                    break
                Fv = Fv_next
                
            # 第二步：水平传播 (Horizontal Propagation) 
            Fh = Fv.copy()
            for i in range(max_iter):
                # 注意：水平传播是在右侧乘以 L_bar 
                Fh_next = alpha * (Fh @ L_bar) + (1 - alpha) * Fv
                if np.linalg.norm(Fh_next - Fh, ord='fro') < tol:
                    Fh = Fh_next
                    break
                Fh = Fh_next
                
            return Fh
        
        else:
            raise ValueError("Method must be 'closed_form' or 'iterative'")

    def adjust_similarity_matrix(self, W, F_star):
        """
        实现论文中的公式 (13): 使用传播后的约束矩阵 F* 调整原始相似度矩阵 W
        
        参数:
        W: np.ndarray, 原始归一化相似度矩阵 (N x N), 元素值在 [0, 1] 之间
        F_star: np.ndarray, 传播后的约束矩阵 (N x N), 元素值在 [-1, 1] 之间
        
        返回:
        W_tilde: np.ndarray, 调整后的新相似度矩阵
        """
        # 确保输入是 numpy 数组
        W = np.asarray(W)
        F_star = np.asarray(F_star)
        
        # 初始化结果矩阵
        W_tilde = np.zeros_like(W)
        
        # 情况 1: f_ij >= 0 (Must-link 倾向)
        # 公式: w_tilde = 1 - (1 - f_ij) * (1 - w_ij)
        mask_pos = (F_star >= 0)
        W_tilde[mask_pos] = 1 - (1 - F_star[mask_pos]) * (1 - W[mask_pos])
        
        # 情况 2: f_ij < 0 (Cannot-link 倾向)
        # 公式: w_tilde = (1 + f_ij) * w_ij
        mask_neg = (F_star < 0)
        W_tilde[mask_neg] = (1 + F_star[mask_neg]) * W[mask_neg]
        
        return W_tilde

    def get_sim_mat(self, X):
        # Cosine similarities, normalized from [-1, 1] to [0, 1]
        M = cosine_similarity(X, X)
        M = (M + 1) / 2
        return M

    def p_pruning(self, A, pval=None):
        if pval is None:
            pval = self.pval
        n_elems = int((1 - pval) * A.shape[0])
        n_elems = min(n_elems, A.shape[0]-self.min_pnum)

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        if k_oracle is None:
            k_oracle = self.k

        lambdas, eig_vecs = scipy.sparse.linalg.eigsh(L, k=min(self.max_num_spks+1, L.shape[0]), which='SM')

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1:self.max_num_spks + 1])
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        # k-means
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    """
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With 
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(self, n_neighbors=20, n_components=60, min_samples=20, min_cluster_size=10, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X, **kwargs):
        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0]-2),
            metric=self.metric,
        ).fit_transform(X)
        labels = hdbscan.HDBSCAN(min_samples=self.min_samples, min_cluster_size=self.min_cluster_size).fit_predict(umap_X)
        return labels

class AHCluster:
    """
    Agglomerative Hierarchical Clustering, a bottom-up approach which iteratively merges 
    the closest clusters until a termination condition is reached.
    This implementation is adapted from https://github.com/BUTSpeechFIT/VBx.
    """

    def __init__(self, fix_cos_thr=0.4):
        self.fix_cos_thr = fix_cos_thr

    def __call__(self, X, **kwargs):
        scr_mx = cosine_similarity(X)
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(scr_mx, method='average', preserve_input='False')
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels = fcluster(lin_mat, -self.fix_cos_thr + adjust, criterion='distance') - 1
        return labels


class CommonClustering:
    """Perfom clustering for input embeddings and output the labels.
    """

    def __init__(self, cluster_type, cluster_line=40, mer_cos=None, min_cluster_size=4, **kwargs):
        self.cluster_type = cluster_type
        self.cluster_line = cluster_line
        self.min_cluster_size = min_cluster_size
        self.mer_cos = mer_cos
        if self.cluster_type == 'spectral':
            self.cluster = SpectralCluster(**kwargs)
        elif self.cluster_type == 'umap_hdbscan':
            kwargs['min_cluster_size'] = min_cluster_size
            self.cluster = UmapHdbscan(**kwargs)
        elif self.cluster_type == 'AHC':
            self.cluster = AHCluster(**kwargs)
        else:
            raise ValueError(
                '%s is not currently supported.' % self.cluster_type
            )
        if self.cluster_type != 'AHC':
            self.cluster_for_short = AHCluster()
        else:
            self.cluster_for_short = self.cluster

    def __call__(self, X, **kwargs):
        # clustering and return the labels
        assert len(X.shape) == 2, 'Shape of input should be [N, C]'
        if X.shape[0] <= 1:
            return np.zeros(X.shape[0], dtype=int)
        if X.shape[0] < self.cluster_line:
            labels = self.cluster_for_short(X)
        else:
            labels = self.cluster(X, **kwargs)

        # remove extremely minor cluster
        labels = self.filter_minor_cluster(labels, X, self.min_cluster_size)
        # merge similar  speaker
        if self.mer_cos is not None:
            labels = self.merge_by_cos(labels, X, self.mer_cos)

        return labels

    def filter_minor_cluster(self, labels, x, min_cluster_size):
        cset = np.unique(labels)
        csize = np.array([(labels == i).sum() for i in cset])
        minor_idx = np.where(csize <= self.min_cluster_size)[0]
        if len(minor_idx) == 0:
            return labels

        minor_cset = cset[minor_idx]
        major_idx = np.where(csize > self.min_cluster_size)[0]
        if len(major_idx) == 0:
            return np.zeros_like(labels)
        major_cset = cset[major_idx]
        major_center = np.stack([x[labels == i].mean(0) \
            for i in major_cset])
        for i in range(len(labels)):
            if labels[i] in minor_cset:
                cos_sim = cosine_similarity(x[i][np.newaxis], major_center)
                labels[i] = major_cset[cos_sim.argmax()]

        return labels

    def merge_by_cos(self, labels, x, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            cset = np.unique(labels)
            if len(cset) == 1:
                break
            centers = np.stack([x[labels == i].mean(0) \
                for i in cset])
            affinity = cosine_similarity(centers, centers)
            affinity = np.triu(affinity, 1)
            idx = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[idx] < cos_thr:
                break
            c1, c2 = cset[np.array(idx)]
            labels[labels==c2]=c1
        return labels


class JointClustering:
    """Perfom joint clustering for input audio and visual embeddings and output the labels.
    """

    def __init__(self, audio_cluster, vision_cluster):
        self.audio_cluster = audio_cluster
        self.vision_cluster = vision_cluster

    def __call__(self, audioX, visionX, audioT, visionT, conf):
        # audio-only and video-only clustering
        alabels = self.audio_cluster(audioX)
        vlabels = self.vision_cluster(visionX)

        alabels = self.arrange_labels(alabels)
        vlist, vspk_embs, vspk_dur = self.get_vlist_embs(audioX, alabels, vlabels, audioT, visionT, conf)

        # modify alabels according to vlabels
        aspk_num = alabels.max()+1
        for i in range(aspk_num):
            aspki_index = np.where(alabels==i)[0]
            aspki_embs = audioX[alabels==i]

            aspkiT_part = np.array(audioT)[alabels==i]
            overlap_vspk = self.overlap_spks(self.cast_overlap(aspkiT_part), vlist, vspk_dur)
            if len(overlap_vspk) > 1:
                centers = np.stack([vspk_embs[s] for s in overlap_vspk])
                distribute_labels = self.distribute_embs(aspki_embs, centers)
                for j in range(distribute_labels.max()+1):
                    for loc in aspki_index[distribute_labels==j]:
                        alabels[loc] = overlap_vspk[j]
            elif len(overlap_vspk) == 1:
                for loc in aspki_index:
                    alabels[loc] = overlap_vspk[0]

        alabels = self.arrange_labels(alabels)
        return alabels

    def overlap_spks(self, times, vlist, vspk_dur=None):
        # get the vspk that overlaps with times.
        overlap_dur = {}
        for [a_st, a_ed] in times:
            for [v_st, v_ed, v_id] in vlist:
                if a_ed > v_st and v_ed > a_st:
                    if v_id not in overlap_dur:
                        overlap_dur[v_id]=0
                    overlap_dur[v_id] += min(a_ed, v_ed) - max(a_st, v_st)
        vspk_list = []
        for v_id, dur in overlap_dur.items():
            # set the criteria for confirming overlap.
            if (vspk_dur is None and dur > 0.5) or (vspk_dur is not None and dur > min(vspk_dur[v_id]*0.5, 0.5)):
                vspk_list.append(v_id)
        return vspk_list

    def distribute_embs(self, embs, centers):
        # embs: [n, D]. centers: [k, D]
        norm_centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        norm_embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        similarity = np.matmul(norm_embs, norm_centers.T) # [n, k]
        argsort = np.argsort(similarity, axis=-1)
        return argsort[:, -1]

    def get_vlist_embs(self, audioX, alabels, vlabels, audioT, visionT, conf):
        assert len(vlabels) == len(visionT)
        vlist = []
        for i, ti in enumerate(visionT):
            if len(vlist)==0 or vlabels[i] != vlist[-1][2] or ti - visionT[i-1] > conf.face_det_stride*0.04 + 1e-4:
                if len(vlist) > 0 and vlist[-1][1] - vlist[-1][0] < 1e-4:
                    # remove too short intervals. 
                    vlist.pop()
                vlist.append([ti, ti, vlabels[i]])
            else:
                vlist[-1][1] = ti

        # adjust vision labels
        vlabels_arrange = self.arrange_labels([i[2] for i in vlist], a_st=alabels.max()+1)
        vlist = [[i[0], i[1], j] for i, j in zip(vlist, vlabels_arrange)]

        # get audio spk embs aligning with 'vlist'
        vspk_embs = {}
        for [v_st, v_ed, v_id] in vlist:
            for i, [a_st, a_ed] in enumerate(audioT):
                if a_ed >= v_st and v_ed >= a_st:
                    if min(a_ed, v_ed) - max(a_st, v_st) > 1:
                        if v_id not in vspk_embs:
                            vspk_embs[v_id] = []
                        vspk_embs[v_id].append(audioX[i])
        for k in vspk_embs:
            vspk_embs[k] = np.stack(vspk_embs[k]).mean(0)

        vlist_new = []
        for i in vlist:
            if i[2] in vspk_embs:
                vlist_new.append(i)
        # get duration of v_spk
        vspk_dur = {}
        for i in vlist_new:
            if i[2] not in vspk_dur:
                vspk_dur[i[2]]=0
            vspk_dur[i[2]] += i[1]-i[0]

        return vlist_new, vspk_embs, vspk_dur

    def cast_overlap(self, input_time):
        if len(input_time)==0:
            return input_time
        output_time = []
        for i in range(0, len(input_time)-1):
            if i == 0 or output_time[-1][1] < input_time[i][0]:
                output_time.append(input_time[i])
            else:
                output_time[-1][1] = input_time[i][1]
        return output_time

    def arrange_labels(self, labels, a_st=0):
        # arrange labels in order from 0.
        new_labels = []
        labels_dict = {}
        idx = a_st
        for i in labels:
            if i not in labels_dict:
                labels_dict[i] = idx
                idx += 1
            new_labels.append(labels_dict[i])
        return np.array(new_labels)
