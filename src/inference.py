# ============================================================
# MACE-X² Research Prototype v0.1
# Local Inference Engine
# ============================================================

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class MACEX2Inference:
    """
    Research-only inference engine.

    Produces:
    - Global cosine 3-NN anomaly score
    - Local 7x7 spatial anomaly score
    - Locked MACE Configuration-1 score

    Not intended for clinical diagnosis.
    """

    def __init__(
        self,
        artifact_dir="artifacts"
    ):

        self.artifact_dir = Path(
            artifact_dir
        )

        # ----------------------------------------------------
        # 1. Device
        # ----------------------------------------------------

        self.device = torch.device("cpu")


        # ----------------------------------------------------
        # 2. Load locked protocols
        # ----------------------------------------------------

        with open(
            self.artifact_dir /
            "mace_configuration1_locked_protocol.json",
            "r"
        ) as f:
            self.protocol = json.load(f)

        with open(
            self.artifact_dir /
            "mace_configuration1_locked_thresholds.json",
            "r"
        ) as f:
            self.thresholds = json.load(f)


        self.global_weight = float(
            self.protocol[
                "selected_global_weight"
            ]
        )

        self.local_weight = float(
            self.protocol[
                "selected_local_weight"
            ]
        )

        self.global_mean = float(
            self.protocol[
                "global_mean"
            ]
        )

        self.global_std = float(
            self.protocol[
                "global_std"
            ]
        )

        self.local_mean = float(
            self.protocol[
                "local_mean"
            ]
        )

        self.local_std = float(
            self.protocol[
                "local_std"
            ]
        )

        self.primary_threshold = float(
            self.thresholds[
                "primary_threshold"
            ]
        )

        self.high_sensitivity_threshold = float(
            self.thresholds[
                "secondary_threshold"
            ]
        )


        # ----------------------------------------------------
        # 3. Load normal global reference embeddings
        # ----------------------------------------------------

        train_features = np.load(
            self.artifact_dir /
            "features.npy"
        ).astype(np.float32)

        assert train_features.shape == (
            13870,
            512
        )

        self.train_reference = (
            torch.from_numpy(
                train_features
            )
        )

        self.train_reference = F.normalize(
            self.train_reference,
            p=2,
            dim=1
        )


        # ----------------------------------------------------
        # 4. Load normal local prototypes
        # ----------------------------------------------------

        local_prototypes = np.load(
            self.artifact_dir /
            "local_normal_position_prototypes.npy"
        ).astype(np.float32)

        assert local_prototypes.shape == (
            49,
            512
        )

        self.local_prototypes = (
            torch.from_numpy(
                local_prototypes
            )
        )

        self.local_prototypes = F.normalize(
            self.local_prototypes,
            p=2,
            dim=1
        )


        # ----------------------------------------------------
        # 5. Frozen ImageNet ResNet18
        # ----------------------------------------------------

        weights = (
            ResNet18_Weights.IMAGENET1K_V1
        )

        self.model = resnet18(
            weights=weights
        )

        self.model.eval()

        for parameter in (
            self.model.parameters()
        ):
            parameter.requires_grad = False


        # ----------------------------------------------------
        # 6. Locked preprocessing
        # ----------------------------------------------------

        self.transform = transforms.Compose([
            transforms.Grayscale(
                num_output_channels=3
            ),

            transforms.Resize(
                (224, 224)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])


    # ========================================================
    # Backbone forward
    # ========================================================

    def _extract_features(
        self,
        image_tensor
    ):

        x = self.model.conv1(
            image_tensor
        )

        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        spatial_map = (
            self.model.layer4(x)
        )

        global_feature = (
            self.model.avgpool(
                spatial_map
            )
        )

        global_feature = torch.flatten(
            global_feature,
            1
        )

        return (
            global_feature,
            spatial_map
        )


    # ========================================================
    # Main prediction
    # ========================================================

    def predict(
        self,
        image
    ):

        if not isinstance(
            image,
            Image.Image
        ):
            raise TypeError(
                "Input must be a PIL Image."
            )


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        image = image.convert("L")

        tensor = self.transform(
            image
        ).unsqueeze(0)


        # ----------------------------------------------------
        # Extract global + local features
        # ----------------------------------------------------

        with torch.inference_mode():

            (
                global_feature,
                spatial_map
            ) = self._extract_features(
                tensor
            )


        # ====================================================
        # GLOBAL WITNESS
        # Locked cosine 3-NN
        # ====================================================

        global_feature = F.normalize(
            global_feature.float(),
            p=2,
            dim=1
        )

        cosine_similarity = (
            global_feature
            @
            self.train_reference.T
        )

        top3_similarity = torch.topk(
            cosine_similarity,
            k=3,
            dim=1,
            largest=True
        ).values.mean(dim=1)

        global_score = float(
            1.0 -
            top3_similarity.item()
        )


        # ====================================================
        # LOCAL WITNESS
        # 49 position-aware spatial scores
        # ====================================================

        spatial_map = F.normalize(
            spatial_map.float(),
            p=2,
            dim=1
        )

        patches = (
            spatial_map
            .permute(
                0,
                2,
                3,
                1
            )
            .reshape(
                1,
                49,
                512
            )
        )

        local_similarity = (
            patches[0]
            *
            self.local_prototypes
        ).sum(dim=1)

        local_anomaly_vector = (
            1.0 -
            local_similarity
        )

        local_score = float(
            local_anomaly_vector
            .mean()
            .item()
        )

        local_map = (
            local_anomaly_vector
            .reshape(7, 7)
            .numpy()
        )


        # ====================================================
        # LOCKED MACE C1 COMBINATION
        # ====================================================

        global_z = (
            global_score -
            self.global_mean
        ) / self.global_std

        local_z = (
            local_score -
            self.local_mean
        ) / self.local_std

        mace_score = (
            self.global_weight *
            global_z
            +
            self.local_weight *
            local_z
        )


        # ====================================================
        # Research evidence category
        # ====================================================

        if (
            mace_score >=
            self.primary_threshold
        ):

            evidence_level = (
                "Higher anomaly evidence"
            )

        elif (
            mace_score >=
            self.high_sensitivity_threshold
        ):

            evidence_level = (
                "Intermediate anomaly evidence"
            )

        else:

            evidence_level = (
                "Lower anomaly evidence"
            )


        return {

            "global_score":
                global_score,

            "local_score":
                local_score,

            "mace_score":
                float(mace_score),

            "local_map":
                local_map,

            "evidence_level":
                evidence_level,

            "primary_threshold":
                self.primary_threshold,

            "high_sensitivity_threshold":
                self.high_sensitivity_threshold
        }


print(
    "✅ STEP 45 COMPLETE — "
    "MACE-X² INFERENCE ENGINE DEFINED"
)