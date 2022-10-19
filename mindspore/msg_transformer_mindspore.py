import mindspore.nn as nn
import mindspore
import numpy as np
from timm_ms import DropPath, to_2tuple, trunc_normal_
from mindspore.common.initializer import initializer



class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.drop = drop
        self.fc1 = nn.Dense(in_features,hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        if (self.drop != 0):
            self.dropout = nn.Dropout(self.drop)

        trunc_normal_(self.fc1.weight, std=.02)
        trunc_normal_(self.fc2.weight, std=.02)
    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if (self.drop != 0):
            x = self.dropout(x)
        x = self.fc2(x)
        if (self.drop != 0):
            x = self.dropout(x)
        return x

def window_partition(x, window_size, shuf_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
        shuf_size (int): shuffle region size

    Returns:
        windows: (B*num_region, shuf_size**2, window_size**2, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size // shuf_size, shuf_size, window_size,
        W // window_size // shuf_size, shuf_size, window_size, C)

    windows = mindspore.ops.transpose(x,(0, 1, 4, 2, 5, 3, 6, 7))
    windows = windows.view(-1, shuf_size**2, window_size**2, C)
    return windows

def window_reverse(windows, window_size, shuf_size, H, W, nchw=False):
    """
    Args:
        windows: (B*num_region, shuf_size**2, window_size**2, C)
        window_size (int): Window size
        shuf_size (int): shuffle region size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size / shuf_size / shuf_size))
    num_region_h = H//window_size//shuf_size
    num_region_w = W//window_size//shuf_size
    x = windows.view(B, num_region_h, num_region_w,
        shuf_size, shuf_size, window_size, window_size, -1)
    if nchw:
        x = mindspore.ops.transpose(x,(0, 7, 1, 3, 5, 2, 4, 6))
        x = x.view(B, -1, H, W)
    else:
        x = mindspore.ops.transpose(x,(0, 1, 3, 5, 2, 4, 6, 7))
        x = x.view(B, H, W, -1)
    return x

def shuffle_msg(x):
    # (B, G, win**2+1, C)
    B, G, N, C = x.shape
    if G == 1:
        return x
    msges = x[:, :, 0] # (B, G, C)
    assert C % G == 0
    msges = msges.view(-1, G, G, C//G)
    msges = mindspore.ops.transpose(msges,(0, 2, 1, 3))
    msges = msges.reshape(B, G, 1, C)
    x = mindspore.ops.concat((msges, x[:, :, 1:]), axis=2)
    return x

class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(WindowAttention,self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # self.relative_position_bias_table = mindspore.Parameter(
        #     np.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        temp = np.zeros(((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        self.relative_position_bias_table = mindspore.Parameter(temp)
        # decouple msg from others for rel pos embed
        self.rel_pos_msg = mindspore.Parameter(np.zeros((num_heads, 2, 1, 1)))

        coords_h = mindspore.Tensor(np.arange(self.window_size[0]))
        coords_w = mindspore.Tensor(np.arange(self.window_size[1]))
        meshgrid = mindspore.ops.Meshgrid()
        grid = meshgrid((coords_h,coords_w))
        coords = mindspore.ops.stack(grid)
        coords_flatten = mindspore.ops.flatten(coords)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = mindspore.ops.transpose(relative_coords,(1, 2, 0))
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)
        self.relative_position_index = relative_position_index

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        if(attn_drop != 0):
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Dense(dim, dim)
        if (proj_drop != 0):
            self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.qkv.weight, std=.02)
        trunc_normal_(self.proj.weight, std=.02)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.rel_pos_msg, std=.02)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """
        Args:
            x: input features with shape of (B, num_windows, N, C)
        """
        B, Ng, N, C = x.shape

        qkv = self.qkv(x).reshape(B, Ng, N, 3, self.num_heads, C // self.num_heads)
        qkv = mindspore.ops.transpose(qkv,(3, 0, 1, 4, 2, 5))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        k = k.transpose(0,1,2,4,3)

        attn = mindspore.numpy.matmul(q,k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)

        rel_pos_others2msg = mindspore.numpy.array(self.rel_pos_msg[:, 0])
        rel_pos_others2msg = mindspore.numpy.repeat(rel_pos_others2msg,self.window_size[0] * self.window_size[1],axis=1)

        rel_pos_msg2others = mindspore.numpy.array(self.rel_pos_msg[:, 1])
        rel_pos_msg2others = mindspore.numpy.repeat(rel_pos_msg2others, (self.window_size[0] * self.window_size[1]+1),axis=2)

        relative_position_bias = mindspore.ops.concat((rel_pos_others2msg, relative_position_bias), axis=-1)
        relative_position_bias = mindspore.ops.concat((rel_pos_msg2others, relative_position_bias), axis=-2)
        expand_dims = mindspore.ops.ExpandDims()
        relative_position_bias = expand_dims(relative_position_bias,0)
        attn += relative_position_bias

        cast = mindspore.ops.Cast()
        attn = cast(attn, mindspore.float32)
        attn = self.softmax(attn)
        if (self.attn_drop != 0):
            attn = self.attn_drop(attn)

        x = mindspore.numpy.matmul(attn, v)

        x = x.transpose(0, 1, 3, 2, 4).reshape(B, Ng, N, C)
        x = self.proj(x)
        if (self.proj_drop != 0):
            x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class MSGBlock(nn.Cell):
    r""" MSG-Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        manip_op: the operation of manipulating msg tokens
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, manip_op=shuffle_msg):
        super(MSGBlock , self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer([dim])
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.manip_op = manip_op


    def construct(self, x):
        cast = mindspore.ops.Cast()
        x = cast(x, mindspore.float32)
        shortcut = x
        x = self.norm1(x)
        # Local-MSA
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        if self.manip_op:
            x = self.manip_op(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        nW = H * W / self.window_size / self.window_size
        # norm1
        flops += self.dim * (H * W + nW)
        # Local-MSA
        flops += nW * self.attn.flops(self.window_size * self.window_size + 1)
        # mlp
        flops += 2 * (H * W + nW) * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * (H * W + nW)
        return flops


class PatchMerging(nn.Cell):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        nxt_shuf_size (int): shuffle region size for the next stage
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, nxt_shuf_size=2):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3,3), stride = 2, pad_mode="pad",padding=1)
        self.norm = norm_layer([2 * dim])
        self.nxt_shuf_size = nxt_shuf_size


    def construct(self, x):
        H, W = self.input_resolution
        B_, shuf_size_2, win_size_2, C = x.shape
        shuf_size = int(shuf_size_2 ** 0.5)
        win_size = int(win_size_2 ** 0.5)
        B = B_ // (H//shuf_size//win_size) // (W//shuf_size//win_size)

        expand_dims = mindspore.ops.ExpandDims()
        temp = expand_dims(x[:, :, 0], 2)
        msg_token = window_reverse(
            temp, 1, shuf_size, H//win_size, W//win_size, nchw=True)

        msg_token = self.reduction(msg_token).transpose(0, 2, 3, 1)
        msg_token = self.norm(msg_token)

        if msg_token.shape[1] >= self.nxt_shuf_size:
            msg_token = window_partition(msg_token, 1, self.nxt_shuf_size)

        x = window_reverse(x[:, :, 1:], win_size, shuf_size, H, W, nchw=True)

        x = self.reduction(x).transpose(0, 2, 3, 1)
        x = self.norm(x)

        if x.shape[1] // win_size >= self.nxt_shuf_size:
            x = window_partition(x, win_size, self.nxt_shuf_size)
        else:
            x = window_partition(x, win_size, 1)
        x = mindspore.ops.concat((msg_token, x), axis=2)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self, win_size):
        H, W = self.input_resolution
        # norm for patch tokens
        flops = H * W * self.dim
        # mlp for patch tokens
        flops += (H // 2) * (W // 2) * 3 * 3 * self.dim * 2 * self.dim
        # norm for msg tokens
        flops += (H // win_size) * (W // win_size) * self.dim
        # mlp for msg tokens
        flops += (H // 2 // win_size) * (W // 2 // win_size) * 3 * 3 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Cell):
    """ A basic MSG-Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        nxt_shuf_size (int): shuffle region size for the next stage
        manip_op: the operation of manipulating msg tokens
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                nxt_shuf_size=2, manip_op=shuffle_msg):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size

        self.blocks = nn.CellList([
            MSGBlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer,
                                manip_op=manip_op)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, nxt_shuf_size=nxt_shuf_size)
        else:
            self.downsample = None

    def construct(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops(self.window_size)
        return flops


class PatchEmbed(nn.Cell):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed,self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(7,7), stride=patch_size, pad_mode="pad", padding=2)

        if norm_layer is not None:
            self.norm = norm_layer([embed_dim])
        else:
            self.norm = None

    def construct(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).transpose(0, 2, 3, 1)  # B Ph Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (7 * 7)
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class MSGTransformer(nn.Cell):
    r""" MSGTransformer
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each MSG-Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        shuffle_size (list(int)): shuffle region size of each stage
        manip_type (str): the operation type for manipulating msg tokens: shuf or none
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                shuffle_size=[4, 4, 2, 1], manip_type='shuf',
                **kwargs):
        super(MSGTransformer,self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.shuffle_size = shuffle_size
        self.drop_rate = drop_rate
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = mindspore.Parameter(np.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        temp = np.zeros((1, shuffle_size[0]**2, 1, embed_dim))
        self.msg_tokens = mindspore.Parameter(temp)
        trunc_normal_(self.msg_tokens, std=.02)

        if(drop_rate != 0):
            self.pos_drop = nn.Dropout(drop_rate)

        # stochastic depth
        start = mindspore.Tensor(0, mindspore.float32)
        stop = mindspore.Tensor(drop_path_rate, mindspore.float32)
        temp = mindspore.ops.linspace(start, stop, sum(depths))
        dpr = []
        for i in range(sum(depths)):
            dpr.append(temp[i])

        if manip_type == 'shuf':
            manip_op = shuffle_msg
        elif manip_type == 'none':
            manip_op = None
        else:
            raise NotImplementedError

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               nxt_shuf_size=shuffle_size[i_layer+1] if (i_layer < self.num_layers - 1) else None,
                               manip_op=manip_op)
            self.layers.append(layer)

        self.norm = norm_layer([self.num_features])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Dense(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)

    def no_weight_decay(self):
        return {'absolute_pos_embed', 'msg_token'}


    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rel_pos_msg'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        if self.ape:
            x = x + self.absolute_pos_embed

        if (self.drop_rate != 0):
            x = self.pos_drop(x)

        x = window_partition(x, self.window_size, self.shuffle_size[0])
        msg_tokens = mindspore.numpy.repeat(self.msg_tokens,x.shape[0],axis=0)
        cast = mindspore.ops.Cast()
        x = cast(x, mindspore.float64)
        x = mindspore.ops.concat((msg_tokens, x), axis=2)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x.squeeze(axis = 1)[:, 0]

    def construct(self, x):

        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * (
            self.patches_resolution[0] * self.patches_resolution[1] // (2 ** (self.num_layers-1))**2 + 1)
        flops += self.num_features * self.num_classes
        return flops


