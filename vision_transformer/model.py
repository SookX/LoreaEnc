import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size=224,
                 path_size = 16,
                 in_channs = 3,
                 embed_dim=768,
                 bias = True
        ):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.path_size = path_size
        self.in_channs = in_channs
        self.embed_dim = embed_dim
        self.num_patches = (img_size // path_size) ** 2

        self.proj = nn.Conv2d(in_channels=in_channs, 
                              out_channels=embed_dim, 
                              kernel_size=path_size, 
                              stride=path_size,
                              bias=bias)
    def forward(self, x):

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x

class SelfAttentionEncoder(nn.Module):
    def __init__(self,
                 embed_dim = 768,
                 num_heads = 12,
                 attn_p = 0.0,
                 proj_p = 0.0,
                 flash_attn = True):
        
        super(SelfAttentionEncoder, self).__init__()
        self.num_heads = num_heads
        self.head_dim = int(embed_dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.flash_attn = flash_attn

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.attn_p = attn_p
        self.attn_drop = nn.Dropout(attn_p)


        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_p)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.q(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.flash_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_p if self.training else 0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class MLP(nn.Module):
    def __init__(self,
                 in_features = 768,
                 mlp_ratio = 4,
                 mlp_p = 0):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(in_features, in_features * mlp_ratio)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(mlp_p)

        self.fc2 = nn.Linear(in_features * mlp_ratio, in_features)
        self.drop2 = nn.Dropout(mlp_p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        return x
    
class EncoderBlock(nn.Module):
    def __init__(self,
                 flash_attn=True,
                 embed_dim = 768,
                 num_heads = 12,
                 mlp_ratio = 4,
                 proj_p = 0.0,
                 attn_p = 0.0,
                 mlp_p = 0.0):
        
        super(EncoderBlock, self).__init__()

        self.norm1= nn.LayerNorm(embed_dim)
        self.attn = SelfAttentionEncoder(embed_dim = embed_dim,
                                         num_heads=num_heads,
                                         attn_p=attn_p,
                                         proj_p=proj_p,
                                         flash_attn=flash_attn)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(in_features=embed_dim,
                       mlp_ratio=mlp_ratio,
                       mlp_p=mlp_p)
        
    def forward(self, x):

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    
class VisionTransformer(nn.Module):
    """
    Vision Transformer as implemented in `An Image is Worth 16x16 Words: Transformer for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16, 
            in_chans=3,
            num_classes=1000,
            flash_attention=True,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            attn_p=0.0,
            mlp_p=0.0, 
            proj_p=0.0,
            pos_p=0.0,
            head_p=0.0,
            pooling="cls", # "cls", "avg"
            custom_weight_init=True,):
        
        super(VisionTransformer, self).__init__()

        self.pooling = pooling
        assert self.pooling in ["cls", "avg"]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            path_size=patch_size,
            in_channs=in_chans,
            embed_dim=embed_dim
        )

        if pooling == "cls":
            num_tokens = self.patch_embed.num_patches + 1
        else:
            num_tokens = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(pos_p)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(flash_attn=flash_attention,
                             embed_dim=embed_dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             proj_p=proj_p,
                             attn_p=attn_p,
                             mlp_p=mlp_p)

                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(head_p)
        self.head = nn.Linear(embed_dim, num_classes)

        if custom_weight_init:
            print("Enabling Custom Weight Initialization")
            self.apply(self._init_weights)
    def _cls_pos_embed(self, x):
        if self.pooling == "cls":
            cls_expand = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_expand, x], dim=1) #prepend the cls token
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self._cls_pos_embed(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)

        if self.pooling == "cls":
            x = x[:, 0]
        
        else:
            x = x.mean(dim=1)
        
        x = self.head_drop(x)
        x = self.head(x)

        return x



if __name__ == "__main__":
    rand = torch.randn(4, 3, 224, 224)
    
    vit = VisionTransformer(pooling="cls")
    vit(rand)

