import numpy as np
import torch


class CBHG(torch.nn.Module):
    """CBHG Module."""

    def __init__(
        self,
        hidden_size,
        K=16,
        projection_size=256,
        num_gru_layers=2,
        max_pool_kernel_size=2,
        is_post=False,
    ):
        """init."""
        # :param hidden_size: dimension of hidden unit
        # :param K: # of convolution banks
        # :param projection_size: dimension of projection unit
        # :param num_gru_layers: # of layers of GRUcell
        # :param max_pool_kernel_size: max pooling kernel size
        # :param is_post: whether post processing or not
        super(CBHG, self).__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.convbank_list = torch.nn.ModuleList()
        self.convbank_list.append(
            torch.nn.Conv1d(
                in_channels=projection_size,
                out_channels=hidden_size,
                kernel_size=1,
                padding=int(np.floor(1 / 2)),
            )
        )

        for i in range(2, K + 1):
            self.convbank_list.append(
                torch.nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=i,
                    padding=int(np.floor(i / 2)),
                )
            )

        self.batchnorm_list = torch.nn.ModuleList()
        for i in range(1, K + 1):
            self.batchnorm_list.append(torch.nn.BatchNorm1d(hidden_size))

        convbank_outdim = hidden_size * K

        self.conv_projection_1 = torch.nn.Conv1d(
            in_channels=convbank_outdim,
            out_channels=hidden_size,
            kernel_size=3,
            padding=int(np.floor(3 / 2)),
        )
        self.conv_projection_2 = torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=projection_size,
            kernel_size=3,
            padding=int(np.floor(3 / 2)),
        )
        self.batchnorm_proj_1 = torch.nn.BatchNorm1d(hidden_size)

        self.batchnorm_proj_2 = torch.nn.BatchNorm1d(projection_size)

        self.max_pool = torch.nn.MaxPool1d(max_pool_kernel_size, stride=1, padding=1)
        self.highway = Highwaynet(self.projection_size)
        self.gru = torch.nn.GRU(
            self.projection_size,
            self.hidden_size // 2,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

    def _conv_fit_dim(self, x, kernel_size=3):
        """_conv_fit_dim."""
        if kernel_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        """forward."""
        input_ = input_.contiguous()
        # batch_size = input_.size(0)
        # total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, batchnorm) in enumerate(
            zip(self.convbank_list, self.batchnorm_list)
        ):
            convbank_input = torch.relu(
                batchnorm(self._conv_fit_dim(conv(convbank_input), k + 1).contiguous())
            )
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:, :, :-1]

        # Projection
        conv_projection = torch.relu(
            self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_cat)))
        )
        conv_projection = (
            self.batchnorm_proj_2(
                self._conv_fit_dim(self.conv_projection_2(conv_projection))
            )
            + input_
        )

        # Highway networks
        highway = self.highway.forward(conv_projection.transpose(1, 2))

        # Bidirectional GRU

        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out


class Highwaynet(torch.nn.Module):
    """Highway network."""

    def __init__(self, num_units, num_layers=4):
        """init."""
        # :param num_units: dimension of hidden unit
        # :param num_layers: # of highway layers

        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.gates = torch.nn.ModuleList()
        self.linears = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.linears.append(torch.nn.Linear(num_units, num_units))
            self.gates.append(torch.nn.Linear(num_units, num_units))

    def forward(self, input_):
        """forward."""
        out = input_

        # highway gated function
        for fc1, fc2 in zip(self.linears, self.gates):

            h = torch.relu(fc1.forward(out))
            t_ = torch.sigmoid(fc2.forward(out))

            c = 1.0 - t_
            out = h * t_ + out * c

        return out
