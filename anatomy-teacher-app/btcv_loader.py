import numpy as np
import cv2
import json
from scipy.sparse import csr_matrix

# Organ labels for BTCV dataset
ORGAN_LABELS = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava",
    10: "Portal/Splenic Vein",
    11: "Pancreas",
    12: "Right Adrenal Gland",
    13: "Left Adrenal Gland"
}


class BTCVDataLoader:
    def __init__(self, btcv_path):
        """
        Load BTCV dataset
        btcv_path: path to BTCV folder (e.g., '../IMIS-Bench-main/dataset/BTCV')
        """
        self.btcv_path = btcv_path

        # Load dataset.json
        with open(f"{btcv_path}/dataset.json", 'r') as f:
            self.dataset_info = json.load(f)

        self.training_data = self.dataset_info['training']
        self.test_data = self.dataset_info['test']

        print(f"✅ Loaded {len(self.training_data)} training images")
        print(f"✅ Loaded {len(self.test_data)} test images")

    def get_question_data(self, index=0, split='training'):
        """Get a specific image and its labels"""
        data_list = self.training_data if split == 'training' else self.test_data

        if index >= len(data_list):
            index = 0

        item = data_list[index]

        # Load image
        image_path = f"{self.btcv_path}/{item['image']}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels (stored as sparse matrix)
        label_path = f"{self.btcv_path}/{item['label']}"
        label_data = np.load(label_path)

        # Reconstruct the sparse matrix
        from scipy.sparse import csr_matrix

        # The npz contains a sparse matrix in CSR format
        sparse_matrix = csr_matrix(
            (label_data['data'], label_data['indices'], label_data['indptr']),
            shape=label_data['shape']
        )

        # Convert to dense array
        labels = sparse_matrix.toarray()

        # Reshape to (13, 512, 512, 1) if needed
        expected_shape = (13, 512, 512, 1)
        if labels.shape != expected_shape:
            # It might be flattened, so reshape it
            labels = labels.reshape(expected_shape)

        print(f"DEBUG: Final labels shape: {labels.shape}")

        return {
            'image': image,
            'labels': labels,
            'image_path': image_path,
            'label_path': label_path
        }

    def get_organ_mask(self, labels, organ_id):
        """
        Extract mask for a specific organ
        labels: numpy array of shape (13, 512, 512, 1)
        organ_id: 1-13 for different organs
        """
        if organ_id < 1 or organ_id > 13:
            raise ValueError("Organ ID must be between 1 and 13")

        # Get the mask for this organ (index is organ_id - 1)
        mask = labels[organ_id - 1, :, :, 0]

        # Convert to binary mask (0 or 1)
        mask = (mask > 0).astype(np.uint8)

        return mask

    def get_random_organ_question(self, index=0):
        """Generate a random question about an organ"""
        import random

        # Get image and labels
        data = self.get_question_data(index)

        # Find which organs are present in this image
        available_organs = []
        for organ_id in range(1, 14):
            mask = self.get_organ_mask(data['labels'], organ_id)
            if mask.sum() > 100:  # Organ must have at least 100 pixels
                available_organs.append(organ_id)

        if not available_organs:
            return None

        # Pick a random organ
        organ_id = random.choice(available_organs)
        organ_name = ORGAN_LABELS[organ_id]

        return {
            'image': data['image'],
            'question': f"Click on the {organ_name}",
            'organ_name': organ_name,
            'organ_id': organ_id,
            'ground_truth_mask': self.get_organ_mask(data['labels'], organ_id),
            'all_labels': data['labels']
        }

    def calculate_dice_score(self, pred_mask, true_mask):
        """
        Calculate Dice similarity coefficient
        1.0 = perfect match, 0.0 = no overlap
        """
        pred_mask = pred_mask.astype(bool)
        true_mask = true_mask.astype(bool)

        intersection = np.logical_and(pred_mask, true_mask).sum()

        if pred_mask.sum() + true_mask.sum() == 0:
            return 0.0

        dice = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum())
        return dice


# Test the loader
if __name__ == "__main__":
    loader = BTCVDataLoader("../IMIS-Bench-main/dataset/BTCV")

    # Get a question
    question = loader.get_random_organ_question(0)

    if question:
        print(f"\n📋 Question: {question['question']}")
        print(f"🎯 Organ: {question['organ_name']}")
        print(f"📐 Ground truth mask shape: {question['ground_truth_mask'].shape}")
        print(
            f"📊 Organ coverage: {question['ground_truth_mask'].sum() / question['ground_truth_mask'].size * 100:.1f}%")