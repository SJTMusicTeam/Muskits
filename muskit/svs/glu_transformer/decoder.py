class Decoder(nn.Module):
    """Decoder Network."""

    def __init__(
        self,
        num_block,
        hidden_size,
        output_dim,
        nhead=4,
        dropout=0.1,
        activation="relu",
        glu_kernel=3,
        local_gaussian=False,
        device="cuda",
    ):
        """init."""
        super(Decoder, self).__init__()
        self.input_norm = module.LayerNorm(hidden_size)
        decoder_layer = module.TransformerGLULayer(
            hidden_size,
            nhead,
            dropout,
            activation,
            glu_kernel,
            local_gaussian=local_gaussian,
            device=device,
        )
        self.decoder = module.TransformerEncoder(decoder_layer, num_block)
        self.output_fc = nn.Linear(hidden_size, output_dim)

        self.hidden_size = hidden_size

    def forward(self, src, pos):
        """forward."""
        if self.training:
            query_mask = pos.ne(0).type(torch.float)
        else:
            query_mask = None
        mask = pos.eq(0).unsqueeze(1).repeat(1, src.size(1), 1)

        src = self.input_norm(src)
        memory, att_weight = self.decoder(src, mask=mask, query_mask=query_mask)
        output = self.output_fc(memory)
        return output, att_weight
