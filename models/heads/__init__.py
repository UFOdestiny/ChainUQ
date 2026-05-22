from models.heads.chainuq_head import ChainUQHead
from models.heads.uq_head_abl_v1 import UQAblationHeadV1
from models.heads.uq_head_abl_v2 import UQAblationHeadV2
from models.heads.uq_head_abl_v3 import UQAblationHeadV3
from models.heads.uq_head_abl_v4 import UQAblationHeadV4

HEAD_REGISTRY = {
    "chainuq": ChainUQHead,
    "uq_abl_v1": UQAblationHeadV1,
    "uq_abl_v2": UQAblationHeadV2,
    "uq_abl_v3": UQAblationHeadV3,
    "uq_abl_v4": UQAblationHeadV4,
}


def build_head(head_type, feature_dim, num_classes=1, **kwargs):
    if head_type not in HEAD_REGISTRY:
        raise ValueError(f"Unknown head: {head_type!r}. Available: {list(HEAD_REGISTRY.keys())}")
    return HEAD_REGISTRY[head_type](feature_dim=feature_dim, num_classes=num_classes, **kwargs)


__all__ = ["HEAD_REGISTRY", "build_head"]
