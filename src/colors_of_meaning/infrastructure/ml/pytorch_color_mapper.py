from typing import List
import numpy.typing as npt
import torch
import torch.nn as nn
from pathlib import Path

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import ColorMapper


class LabProjectorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, 3),
        )

        self.l_activation = nn.Sigmoid()
        self.ab_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)

        lightness = self.l_activation(features[:, 0:1]) * 100.0
        a_val = self.ab_activation(features[:, 1:2]) * 127.5
        b_val = self.ab_activation(features[:, 2:3]) * 127.5

        return torch.cat([lightness, a_val, b_val], dim=1)


class PyTorchColorMapper(ColorMapper):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.network = LabProjectorNetwork(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def embed_to_lab(self, embedding: npt.NDArray) -> LabColor:
        self.network.eval()
        with torch.no_grad():
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            lab_tensor = self.network(embedding_tensor)
            lab_values = lab_tensor.cpu().numpy()[0]

        return LabColor(l=float(lab_values[0]), a=float(lab_values[1]), b=float(lab_values[2])).clamp()

    def embed_batch_to_lab(self, embeddings: npt.NDArray) -> List[LabColor]:
        self.network.eval()
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            lab_tensor = self.network(embeddings_tensor)
            lab_values = lab_tensor.cpu().numpy()

        return [LabColor(l=float(row[0]), a=float(row[1]), b=float(row[2])).clamp() for row in lab_values]

    def train(self, embeddings: npt.NDArray, epochs: int, learning_rate: float) -> None:
        self.network.train()

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        targets = self._generate_targets(embeddings_tensor)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        batch_size = min(32, len(embeddings))
        num_batches = (len(embeddings) + batch_size - 1) // batch_size

        for epoch in range(epochs):
            total_loss = 0.0
            indices = torch.randperm(len(embeddings))

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(embeddings))
                batch_indices = indices[start_idx:end_idx]

                batch_embeddings = embeddings_tensor[batch_indices]
                batch_targets = targets[batch_indices]

                optimizer.zero_grad()
                predictions = self.network(batch_embeddings)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def save_weights(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.network.eval()

    @staticmethod
    def _generate_targets(embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = embeddings.shape[0]

        l_values = torch.rand(batch_size, 1) * 100.0
        a_values = torch.rand(batch_size, 1) * 255.0 - 128.0
        b_values = torch.rand(batch_size, 1) * 255.0 - 128.0

        return torch.cat([l_values, a_values, b_values], dim=1)
